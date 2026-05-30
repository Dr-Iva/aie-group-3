from src.extraction.value_extractor import ValueExtractor


def test_extract_voltage_range() -> None:
    extractor = ValueExtractor()
    text = """
    Absolute Maximum Ratings
    Supply voltage: 1.8V to 6.0V.
    """
    result = extractor.extract("Максимальное напряжение питания MCP6002", text)

    assert result is not None
    assert result.unit == "V"
    assert result.value == "1.8 to 6.0 V" or result.value == "1.8V to 6.0V" or "1.8" in result.value


def test_extract_temperature_range() -> None:
    extractor = ValueExtractor()
    text = "Operating temperature range: -40°C to +125°C."
    result = extractor.extract("рабочая температура", text)

    assert result is not None
    assert result.unit == "°C"
    assert "40" in result.value
