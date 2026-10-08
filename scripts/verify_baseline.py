"""Verify real baseline bundles after training, without opening held-out data."""

from pathlib import Path

import numpy as np

from ticket_intelligence.inference.predictor import Predictor


def main() -> None:
    predictor = Predictor.from_artifacts(
        Path("artifacts/baselines"), Path("configs/team_mapping.json")
    )
    samples = [
        "My card payment was declined at the shop today.",
        "The ATM took my card and did not return it.",
        "I forgot the passcode for my account.",
        "My recipient has still not received the transfer I sent.",
        "zzzxxyy qqqwwwvv",
    ]
    probabilities = predictor.predict_probabilities(samples)
    assert probabilities.shape == (len(samples), 77)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    for text in samples:
        print(text)
        print(predictor.predict(text))
    print("Reloaded real model; all 77 probability columns are valid.")


if __name__ == "__main__":
    main()
