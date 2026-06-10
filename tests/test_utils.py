from hr_predictor.utils import american_to_implied_probability, confidence_tier


def test_american_to_implied_probability_positive_odds():
    assert round(american_to_implied_probability(300), 4) == 0.25


def test_american_to_implied_probability_negative_odds():
    assert round(american_to_implied_probability(-150), 4) == 0.6


def test_confidence_tier():
    assert confidence_tier(0.13) == "A"
    assert confidence_tier(0.09) == "B"
    assert confidence_tier(0.06) == "C"
    assert confidence_tier(0.03) == "Longshot"

