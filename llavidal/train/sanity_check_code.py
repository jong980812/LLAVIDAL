import logging
from contextlib import nullcontext

def count_params_proper(model):
    """ZeRO-3/가중치공유/메타텐서까지 고려한 정확한 카운트"""
    try:
        import deepspeed
        zero = deepspeed.zero
    except Exception:
        zero = None

    seen = set()
    total = 0
    trainable = 0

    def _numel_full(p):
        # ZeRO-3일 때 전체 파라미터 크기 복원
        if zero is not None and hasattr(p, "ds_id"):
            ctx = zero.GatheredParameters([p], modifier_rank=None, enabled=True)
        else:
            ctx = nullcontext()
        with ctx:
            return p.numel()

    for p in model.parameters():
        # 가중치 공유(tied) 중복 제거
        pid = p.data_ptr() if p.device.type != "meta" else id(p)
        if pid in seen:
            continue
        seen.add(pid)

        n = _numel_full(p)
        total += n
        if p.requires_grad:
            trainable += n

    return total, trainable, total - trainable

def fmt(n):
    return f"{n/1e9:.2f}B" if n >= 1e9 else f"{n/1e6:.2f}M"

def log_param_counts(model, topk=10):
    # DeepSpeed Engine이면 내부 원모델로
    if hasattr(model, "module"):
        model_ = model.module
    else:
        model_ = model

    total, trainable, frozen = count_params_proper(model_)
    logging.warning(
        "Parameter count (global):\n"
        f"- TOTAL     : {fmt(total)} ({total:,})\n"
        f"- TRAINABLE : {fmt(trainable)} ({trainable:,})\n"
        f"- FROZEN    : {fmt(frozen)} ({frozen:,})"
    )

    # 학습 가능한 상위 텐서 몇 개 보기
    contrib = []
    seen = set()
    for n, p in model_.named_parameters():
        pid = p.data_ptr() if p.device.type != "meta" else id(p)
        if not p.requires_grad or pid in seen:
            continue
        seen.add(pid)
        # ZeRO-3 고려
        try:
            import deepspeed
            with deepspeed.zero.GatheredParameters([p], modifier_rank=None, enabled=hasattr(p, "ds_id")):
                contrib.append((n, p.numel()))
        except Exception:
            contrib.append((n, p.numel()))
    contrib.sort(key=lambda x: x[1], reverse=True)
    if contrib:
        logging.warning("Top trainable tensors:")
        for name, n in contrib[:topk]:
            logging.warning(f"  - {name:<60} {fmt(n)} ({n:,})")
