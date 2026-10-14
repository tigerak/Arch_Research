def extract_model_state_dict(weight_obj):
    """체크포인트 호환 로더.

    새 포맷(dict: model_state_dict/arcface_state_dict/...)과
    구 포맷(raw state_dict) 둘 다 지원한다.
    """
    if isinstance(weight_obj, dict) and 'model_state_dict' in weight_obj:
        return weight_obj['model_state_dict']
    return weight_obj
