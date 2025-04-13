import torch
from torch.nn import Module
import utils.printing as printing
from typing import List, Tuple
import abc
import torch.nn.functional as F
from jaxtyping import Tensor
from transformers.cache_utils import DynamicCache
from typing import Union
from termcolor import colored

def token_ids_to_string(token_ids, tokenizer):
    """Convert token ids to string.

    Args:
        token_ids (List[int]): List of token ids.
        tokenizer (Tokenizer): Tokenizer.

    Returns:
        str: String representation of token ids.
    """
    strings = tokenizer.convert_ids_to_tokens(token_ids)
    return " ".join(strings)


def end_token_found(location: int):
    print(colored(f"End token found at position {location}", "red"))


def initial_step(token: Tensor, tokenizer):
    print(
        f"{colored('Initiale Step', on_color='on_dark_grey', color='white')} 1 token:"
    )
    print(colored(token_ids_to_string(token, tokenizer), "blue"))


def speculative_step(
    tokenizer,
    current_inputs: Tensor,
    inputs: Tensor,
    n: int,
    prompt_end: int,
    current_position: int,
    corrected_gamma: int,
):
    print(
        f"{colored('Speculative Step', on_color='on_dark_grey', color='white')} {n} draft{'s' if n > 1 else ''} + 1 token:"
    )
    print(
        token_ids_to_string(inputs[0, prompt_end:current_position], tokenizer), end=" "
    )
    print(
        colored(
            token_ids_to_string(
                inputs[0, current_position : current_position + n], tokenizer
            ),
            "green",
        ),
        end=(" " if n > 0 else ""),
    )
    print(
        colored(
            token_ids_to_string(
                current_inputs[
                    0, current_position + n : current_position + corrected_gamma
                ],
                tokenizer,
            ),
            "red",
        ),
        end=(" " if n < corrected_gamma else ""),
    )
    print(
        colored(
            token_ids_to_string(inputs[..., current_position + n], tokenizer), "blue"
        )
    )


def beam_search_step(
    possibilities: List[Tuple[float, Tensor, Tensor]], current_position: int, tokenizer
):
    print(
        f"{colored('Beam Search Step', on_color='on_dark_grey', color='white')} Token {current_position}:"
    )

    for i, (prob, tokens, _) in enumerate(possibilities):
        print(
            f"{i + 1}. {prob:.3f}\t{token_ids_to_string(tokens[: current_position - 1], tokenizer)} {colored(token_ids_to_string(tokens[current_position - 1 : current_position], tokenizer), 'blue')}"
        )


class LogitsProcessor(abc.ABC):
    """Logits processors for sampling."""

    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, logits: Tensor) -> Tensor:
        proc = self._process(logits)
        return F.softmax(proc / self.temperature, dim=-1)

    @abc.abstractmethod
    def _process(self, logits: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def sample(self, probs: Tensor) -> Tensor:
        pass


class GreedyProcessor(LogitsProcessor):
    """Greedy: Most probable token."""

    def __init__(self, temperature: float = 1):
        super().__init__(temperature)

    def _process(self, logits: Tensor) -> Tensor:
        return logits

    def sample(self, probs: Tensor) -> Tensor:
        return torch.argmax(probs, dim=-1).unsqueeze(-1)


def prune_cache(
    cache: Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache], num_tokens_to_discard: int
):
    """
    Prune the cache by removing the specified number of tokens from the end.

    Args:
        cache (Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache]): The KV cache to be pruned.
        num_tokens_to_discard (int): The number of tokens to discard from the end of the cache.

    Returns:
        Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache]: The pruned KV cache.
    """
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return prune_tuple_cache(cache, num_tokens_to_discard)
    elif isinstance(cache, DynamicCache):
        return prune_dynamic_cache(cache, num_tokens_to_discard)
    else:
        raise ValueError("Unsupported cache type.")


def max_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Max function.
        x: input tensor.
    Returns:
        tensor norm(max(0, x)).
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum


@torch.no_grad()
def speculative_generate(
    inputs: List[int],
    drafter: Module,
    target: Module,
    tokenizer=None,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    skip_sample_adjustment: bool = False,
    first_target: bool = True,
    debug: bool = False,
) -> Tuple[List[int], float]:
    """
    Generate text sequence using the speculative decoding algorithm.
    Implementation of Speculative Decoding. (https://arxiv.org/pdf/2211.17192.pdf)

    Args:
        inputs (List[int]): input sequence of batch size 1.
        drafter (Module): drafter model.
        target (Module): target model.
        tokenizer: tokenizer (used for debugging).
        gamma (int): number of drafts generated by the drafter at each step.
        logits_processor (LogitsProcessor): logits processor for sampling.
        max_gen_len (int): maximum length of the generated sequence.
        eos_tokens_id (int or List[int]): end token id (could be multiple).
        pad_token_id (int): pad token id.
        use_cache (bool): whether to use cache.
        skip_sample_adjustment (bool): whether to skip the sample adjustment step when some drafts are discarded.
        first_target (bool): whether to run the target model before the speculative algorithm.
        debug (bool): debug mode.

    Returns:
        List[int]: generated sequence.
        float: acceptance rate (number of accepted drafts divided by the number of total drafts).

    Note: This generation methods only works for decoder-only models.
    Note bis: The drafter and target models should output the same logits shape.
    Note ter: NgramModels are currently not supported.
    """

    drafter_cache, target_cache = None, None

    list_tokens_id = (
        eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    )
    stop_tokens = torch.tensor(
        list_tokens_id, dtype=torch.long, device=target.device
    ).unsqueeze(1)

    drafts_accepted, drafts_speculated = 0.0, 0.0

    vocabulary_size = target.config.vocab_size

    # prepare input tensor
    prompt_len = len(inputs)
    max_seq_length = (
        target.config.max_position_embeddings
        if hasattr(target.config, "max_position_embeddings")
        else (
            target.config.max_context_length
            if hasattr(target.config, "max_context_length")
            else 1024
        )
    )
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = torch.full(
        (1, total_len), pad_token_id, dtype=torch.long, device=target.device
    )
    input_ids[0, :prompt_len] = torch.tensor(
        inputs, dtype=torch.long, device=target.device
    )

    current_position = prompt_len

    if first_target:
        # run the target model before the speculative algorithm. Allows to prefill the kvcache and get a first token.
        Mp = target(
            input_ids=input_ids[..., :current_position],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values
        p_p = logits_processor(Mp.logits[..., -1, :])
        t = logits_processor.sample(p_p)
        input_ids[0, current_position] = t
        current_position += 1

        if torch.isin(t, stop_tokens):
            if debug:
                end_token_found(0)
            return input_ids[0, prompt_len:current_position].tolist(), 0

        if debug:
            initial_step(t, tokenizer)

    while current_position < total_len:
        corrected_gamma = min(gamma, total_len - current_position - 1)
        q = torch.zeros((1, corrected_gamma, vocabulary_size), device=target.device)

        input_ids = input_ids.to(drafter.device)

        # generate gamma drafts
        for k in range(corrected_gamma):
            Mq = drafter(
                input_ids=input_ids[..., : current_position + k],
                past_key_values=drafter_cache,
                use_cache=use_cache,
            )
            drafter_cache = Mq.past_key_values

            draft_logits = Mq.logits[..., -1, :]
            draft_probs = logits_processor(draft_logits)
            q[0, k] = draft_probs.to(target.device)
            xi = logits_processor.sample(draft_probs)
            input_ids[0, current_position + k] = xi
        drafts_speculated += corrected_gamma
        input_ids = input_ids.to(target.device)

        # run target model on drafts and get logits of the previous tokens plus one more token
        Mp = target(
            input_ids=input_ids[..., : current_position + corrected_gamma],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values
        draft_logits = Mp.logits[
            ..., current_position - 1 : current_position + corrected_gamma - 1, :
        ]  # [1, corrected_gamma, vocab_size]
        p = logits_processor(draft_logits)  # [1, gamma, vocab_size]

        # compute the last accepted draft position (rejection sampling)
        r = torch.rand(corrected_gamma, device=target.device)
        fractions = p / q
        n = corrected_gamma
        for i in range(corrected_gamma):
            if r[i] > fractions[0, i, input_ids[0, current_position + i]]:
                n = i
                break

        drafts_accepted += n

        # check if the end token is in the drafts
        stop_locations = torch.nonzero(
            torch.eq(
                input_ids[..., current_position : current_position + n], stop_tokens
            )
        )
        if stop_locations.shape[0] > 0:
            stop_location = stop_locations[0, 1].item()
            if debug:
                end_token_found(stop_location)
            return input_ids[
                0, prompt_len : current_position + stop_location + 1
            ].tolist(), drafts_accepted / drafts_speculated

        # adjust the distribution from Mp
        if n == corrected_gamma:
            p_p = Mp.logits[..., current_position + corrected_gamma - 1, :]
            p_p = logits_processor(p_p)
        else:
            # prune the cache
            if use_cache:
                drafter_cache = prune_cache(drafter_cache, corrected_gamma - n)
                target_cache = prune_cache(target_cache, corrected_gamma - n + 1)

            if not skip_sample_adjustment:
                p_p = max_fn(p[..., n, :] - q[0, n, :])
            else:
                p_p = p[..., n, :]
        x = logits_processor.sample(p_p)

        if debug:
            generated = input_ids.clone().detach()

        input_ids[0, current_position + n : current_position + corrected_gamma] = (
            pad_token_id
        )
        input_ids[0, current_position + n] = x

        if debug:
            speculative_step(
                tokenizer,
                generated,
                input_ids,
                n,
                prompt_len,
                current_position,
                corrected_gamma,
            )

        current_position += n + 1

        if torch.isin(x, stop_tokens):
            if debug:
                end_token_found(n)
            return input_ids[
                0, prompt_len:current_position
            ].tolist(), drafts_accepted / drafts_speculated

    return input_ids[0, prompt_len:].tolist(), drafts_accepted / drafts_speculated
