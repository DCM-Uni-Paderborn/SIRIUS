! This file is part of SIRIUS electronic structure library.
! Copyright (c), ETH Zurich. All rights reserved.
! SPDX-License-Identifier: BSD-3-Clause

module sirius_testing
use, intrinsic :: ISO_C_BINDING, only: C_PTR, C_DOUBLE, C_INT
use sirius, only: sirius_ground_state_handler, sirius_get_ground_state_handle
implicit none
private
public :: sirius_check_external_xc_derivative, sirius_check_external_xc_response

interface
    integer(C_INT) function check_derivative(handle, step, values) &
            bind(C, name="sirius_test_external_xc_derivative")
        import C_PTR, C_DOUBLE, C_INT
        type(C_PTR), value :: handle
        real(C_DOUBLE), value :: step
        real(C_DOUBLE), intent(out) :: values(3, 2)
    end function

    integer(C_INT) function check_response(handle, step, errors) &
            bind(C, name="sirius_test_external_xc_response")
        import C_PTR, C_DOUBLE, C_INT
        type(C_PTR), value :: handle
        real(C_DOUBLE), value :: step
        real(C_DOUBLE), intent(out) :: errors(2)
    end function
end interface

contains

!> Orbital derivatives at fixed geometry, occupations and core states.
subroutine sirius_check_external_xc_derivative(gs_handler, step, derivatives, error_code)
    type(sirius_ground_state_handler), intent(in) :: gs_handler
    real(C_DOUBLE), intent(in) :: step
    real(C_DOUBLE), intent(out) :: derivatives(3, 2)
    integer, intent(out) :: error_code
    error_code = check_derivative(sirius_get_ground_state_handle(gs_handler), step, derivatives)
end subroutine

!> XC field and enabled point-input response checks without changing the ground state.
subroutine sirius_check_external_xc_response(gs_handler, step, errors, error_code)
    type(sirius_ground_state_handler), intent(in) :: gs_handler
    real(C_DOUBLE), intent(in) :: step
    real(C_DOUBLE), intent(out) :: errors(2)
    integer, intent(out) :: error_code
    error_code = check_response(sirius_get_ground_state_handle(gs_handler), step, errors)
end subroutine

end module sirius_testing
