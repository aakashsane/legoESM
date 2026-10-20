"""Card-selected NEMO switches must fail loudly instead of reading as off."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import surface_stress_faces
from legoesm.ocean.vertical import static_card_flag


def test_static_card_flag_reads_concrete_values():
    assert static_card_flag(SimpleNamespace(f=True), "f") is True
    assert static_card_flag(SimpleNamespace(f=False), "f") is False
    assert static_card_flag(SimpleNamespace(), "f") is False


def test_static_card_flag_refuses_a_traced_switch():
    @jax.jit
    def read(flag):
        return static_card_flag(SimpleNamespace(f=flag), "f")

    with pytest.raises(TypeError, match="traced"):
        read(True)


def test_fold_vector_sign_refuses_geographic_stress():
    forcing = SimpleNamespace(tau_x=jnp.zeros((2, 2)), tau_y=jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match="tau_i_native"):
        surface_stress_faces(forcing, jnp.float64, None, None, None,
                             nemo_fold_vector_sign=True)


def test_fold_vector_sign_without_any_stress_is_still_none():
    assert surface_stress_faces(SimpleNamespace(), jnp.float64, None, None,
                                None, nemo_fold_vector_sign=True) is None
