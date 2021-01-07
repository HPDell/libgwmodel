from regression_diagnostic cimport GwmRegressionDiagnostic

cdef class RegressionDiagnostic:
    def __cinit__(self, double rss, double aic, double aic_c, double enp, double edf, double r_square, double r_square_adjust):
        self._c_instance = GwmRegressionDiagnostic(rss, aic, aic_c, enp, edf, r_square, r_square_adjust)
    
    def __dealloc__(self):
        self._c_instance.RSS = 0.0
        self._c_instance.AIC = 0.0
        self._c_instance.AICc = 0.0
        self._c_instance.ENP = 0.0
        self._c_instance.EDF = 0.0
        self._c_instance.RSquare = 0.0
        self._c_instance.RSquareAdjust = 0.0

    @staticmethod
    cdef RegressionDiagnostic wrap(GwmRegressionDiagnostic instance):
        cdef RegressionDiagnostic wrapper = RegressionDiagnostic(
            instance.RSS,
            instance.AIC,
            instance.AICc,
            instance.ENP,
            instance.EDF,
            instance.RSquare,
            instance.RSquareAdjust
        )
        return wrapper
    
    @property    
    def rss(self):
        return self._c_instance.RSS

    @property
    def aic(self):
        return self._c_instance.AIC

    @property
    def aic_c(self):
        return self._c_instance.AICc

    @property
    def enp(self):
        return self._c_instance.ENP

    @property
    def edf(self):
        return self._c_instance.EDF

    @property
    def r_square(self):
        return self._c_instance.RSquare

    @property
    def r_square_adjust(self):
        return self._c_instance.RSquareAdjust
