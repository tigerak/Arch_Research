def extract_model_state_dict(weight_obj):
    """체크포인트를 model_state_dict 형태로 정규화한다.

    이 함수는 여러 체크포인트 형식을 안전하게 흡수하기 위한 helper이다.

    지원 포맷:
    - {"model_state_dict": {...}}
    - {"state_dict": {...}}
    - raw state_dict 딕셔너리

    실험 포털에서는 여러 모델/메서드에서 체크포인트를 다루기 때문에,
    저장 형식이 조금씩 달라도 공통 인터페이스로 정규화하는 것이 중요하다.
    """
    if isinstance(weight_obj, dict) and 'model_state_dict' in weight_obj:
        return weight_obj['model_state_dict']
    return weight_obj
