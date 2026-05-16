"""
Quick test to verify backward compatibility of distributional RL implementation.

This script tests that:
1. Default config (all flags False) works identically to original
2. Models can be created with distributional critic enabled
3. Basic forward passes work correctly
"""

import sys
import torch
import numpy as np

# Import the modified modules
from pyrl import configs
from pyrl.policygradient_mc import PolicyGradient
from pyrl.performance import Performance2AFC
import tasks.gambling as gambling_task

def test_config_defaults():
    """Test that all new config parameters have safe defaults."""
    print("\n=== Test 1: Config Defaults ===")

    required_flags = [
        'use_distributional_critic',
        'use_context_quantile_selection',
        'use_context_temperature',
    ]

    for flag in required_flags:
        if flag not in configs.default:
            print(f"❌ FAIL: Missing config flag: {flag}")
            return False
        if configs.default[flag] != False:
            print(f"❌ FAIL: Config flag {flag} should default to False, got {configs.default[flag]}")
            return False

    print("✅ PASS: All new config flags default to False (backward compatible)")
    return True


def test_original_mode():
    """Test that original (non-distributional) mode still works."""
    print("\n=== Test 2: Original Mode (All Flags Off) ===")

    try:
        # Create minimal config for gambling task
        # Use a simple dummy task class
        class DummyTask:
            def get_condition(self, rng, dt):
                return {
                    'durations': {'fixation': (0, 250), 'tmax': 760},
                    'time': np.arange(0, 760, dt),
                    'epochs': {'fixation': range(25)}
                }

            def get_step(self, rng, dt, trial, t, a):
                u = np.zeros(7)
                r = 0.0
                status = {'continue': t < 75, 'reward': r}
                return u, r, status

        # Start with defaults and override
        config = {**configs.default}
        config.update({
            'inputs': gambling_task.inputs,
            'actions': gambling_task.actions,
            'Nin': len(gambling_task.inputs),  # Number of inputs
            'Nout': len(gambling_task.actions),  # Number of outputs
            'tmax': 760,
            'n_gradient': 2,
            'n_validation': 2,
            'N': 50,
            'baseline_N': 50,
            'max_iter': 1,
            'Performance': Performance2AFC,
            # Explicit: all distributional flags OFF
            'use_distributional_critic': False,
            'use_context_quantile_selection': False,
            'use_context_temperature': False,
        })

        # Create PolicyGradient instance
        pg = PolicyGradient(DummyTask, config, seed=1, device='cpu')

        # Check baseline output size (should be 1)
        if pg.baseline_net.Nout != 1:
            print(f"❌ FAIL: Baseline Nout should be 1 in original mode, got {pg.baseline_net.Nout}")
            return False

        # Check that distributional flag is off
        if pg.use_distributional:
            print(f"❌ FAIL: use_distributional should be False")
            return False

        # Run a single trial to test forward pass
        rng = np.random.RandomState(1)
        trial = pg.task.get_condition(rng, config['dt'])
        results = pg.run_trials([trial], return_states=True)

        # Check Z_b shape (should be (T, B) not (T, B, n_quantiles))
        Z_b = results['Z_b']
        if len(Z_b.shape) != 2:
            print(f"❌ FAIL: Z_b should have shape (T, B), got {Z_b.shape}")
            return False

        print("✅ PASS: Original mode works correctly")
        print(f"  - Baseline Nout: {pg.baseline_net.Nout}")
        print(f"  - Z_b shape: {Z_b.shape}")
        return True

    except Exception as e:
        print(f"❌ FAIL: Exception in original mode: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distributional_mode():
    """Test that distributional mode can be enabled and runs."""
    print("\n=== Test 3: Distributional Mode (Flag On) ===")

    try:
        # Use the same dummy task
        class DummyTask:
            def get_condition(self, rng, dt):
                return {
                    'durations': {'fixation': (0, 250), 'tmax': 760},
                    'time': np.arange(0, 760, dt),
                    'epochs': {'fixation': range(25)}
                }

            def get_step(self, rng, dt, trial, t, a):
                u = np.zeros(7)
                r = 0.0
                status = {'continue': t < 75, 'reward': r}
                return u, r, status

        # Create config with distributional critic enabled
        config = {**configs.default}
        config.update({
            'inputs': gambling_task.inputs,
            'actions': gambling_task.actions,
            'Nin': len(gambling_task.inputs),  # Number of inputs
            'Nout': len(gambling_task.actions),  # Number of outputs
            'tmax': 760,
            'n_gradient': 2,
            'n_validation': 2,
            'N': 50,
            'baseline_N': 50,
            'max_iter': 1,
            'Performance': Performance2AFC,
            # Distributional mode ON
            'use_distributional_critic': True,
            'n_quantiles': 5,
            'quantile_huber_kappa': 1.0,
            'use_context_quantile_selection': False,  # Start simple
            'use_context_temperature': False,
        })

        # Create PolicyGradient instance
        pg = PolicyGradient(DummyTask, config, seed=1, device='cpu')

        # Check baseline output size (should be 5)
        if pg.baseline_net.Nout != 5:
            print(f"❌ FAIL: Baseline Nout should be 5 in distributional mode, got {pg.baseline_net.Nout}")
            return False

        # Check that distributional flag is on
        if not pg.use_distributional:
            print(f"❌ FAIL: use_distributional should be True")
            return False

        # Check quantile values
        if not hasattr(pg, 'tau_values'):
            print(f"❌ FAIL: Missing tau_values attribute")
            return False

        if len(pg.tau_values) != 5:
            print(f"❌ FAIL: tau_values should have 5 elements, got {len(pg.tau_values)}")
            return False

        # Run a single trial
        rng = np.random.RandomState(1)
        trial = pg.task.get_condition(rng, config['dt'])
        results = pg.run_trials([trial], return_states=True)

        # Check Z_b shape (should be (T, B, n_quantiles))
        Z_b = results['Z_b']
        if len(Z_b.shape) != 3:
            print(f"❌ FAIL: Z_b should have shape (T, B, n_quantiles), got {Z_b.shape}")
            return False

        if Z_b.shape[2] != 5:
            print(f"❌ FAIL: Z_b should have 5 quantiles, got {Z_b.shape[2]}")
            return False

        print("✅ PASS: Distributional mode works correctly")
        print(f"  - Baseline Nout: {pg.baseline_net.Nout}")
        print(f"  - Z_b shape: {Z_b.shape}")
        print(f"  - Tau values: {pg.tau_values.cpu().numpy()}")
        return True

    except Exception as e:
        print(f"❌ FAIL: Exception in distributional mode: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantile_utilities():
    """Test distributional utility functions."""
    print("\n=== Test 4: Distributional Utilities ===")

    try:
        from pyrl.distributional_utils import (
            quantile_huber_loss,
            interpolate_quantiles,
            context_to_quantile_idx,
            get_default_quantiles
        )

        # Test get_default_quantiles
        tau = get_default_quantiles(5)
        if len(tau) != 5:
            print(f"❌ FAIL: get_default_quantiles(5) should return 5 values, got {len(tau)}")
            return False

        # Test context_to_quantile_idx
        context = torch.tensor([1.0])  # Risk-seeking
        idx = context_to_quantile_idx(context, 5)
        if not (idx > 2.0).all():  # Should map to upper quantiles
            print(f"❌ FAIL: Risk-seeking context should map to high quantile index, got {idx}")
            return False

        # Test interpolate_quantiles
        q_values = torch.randn(10, 2, 5)  # (T, B, n_quantiles)
        quantile_idx = torch.tensor([2.5, 2.5])  # Middle of 3rd and 4th quantile
        interpolated = interpolate_quantiles(q_values, quantile_idx)
        if interpolated.shape != (10, 2):
            print(f"❌ FAIL: interpolate_quantiles should return (T, B), got {interpolated.shape}")
            return False

        # Test quantile_huber_loss
        pred_quantiles = torch.randn(10, 2, 5)  # (T, B, n_quantiles)
        targets = torch.randn(10, 2)  # (T, B)
        tau_values = get_default_quantiles(5)
        loss = quantile_huber_loss(pred_quantiles, targets, tau_values, kappa=1.0)
        if not torch.isfinite(loss):
            print(f"❌ FAIL: quantile_huber_loss returned non-finite value: {loss}")
            return False

        print("✅ PASS: All distributional utilities work correctly")
        print(f"  - Default quantiles: {tau.numpy()}")
        print(f"  - Context mapping works")
        print(f"  - Quantile interpolation works")
        print(f"  - Quantile Huber loss computes: {loss.item():.4f}")
        return True

    except Exception as e:
        print(f"❌ FAIL: Exception in utilities test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print(" Backward Compatibility Test Suite")
    print("="*60)

    results = []

    # Run tests
    results.append(("Config Defaults", test_config_defaults()))
    results.append(("Original Mode", test_original_mode()))
    results.append(("Distributional Mode", test_distributional_mode()))
    results.append(("Utility Functions", test_quantile_utilities()))

    # Summary
    print("\n" + "="*60)
    print(" Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Backward compatibility verified.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. See details above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
