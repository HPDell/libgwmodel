from cbase cimport GwmRegressionDiagnostic

cdef class RegressionDiagnostic:
    cdef GwmRegressionDiagnostic _c_instance

    @staticmethod
    cdef RegressionDiagnostic wrap(GwmRegressionDiagnostic instance)