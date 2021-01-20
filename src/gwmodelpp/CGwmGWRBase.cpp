#include "CGwmGWRBase.h"
#include <assert.h>

CGwmGWRBase::CGwmGWRBase()
{
    
}

CGwmGWRBase::~CGwmGWRBase()
{

}

bool CGwmGWRBase::isValid()
{
    if (CGwmSpatialMonoscaleAlgorithm::isValid())
    {
        if (mIndepVars.size() < 1)
            return false;

        return true;
    }
    else return false;
}

void CGwmGWRBase::setXY(mat& x, mat& y, const CGwmSimpleLayer* layer, const GwmVariable& depVar, const vector<GwmVariable>& indepVars)
{
    uword nDp = layer->featureCount(), nVar = indepVars.size() + 1;
    arma::uvec indepVarIndeces(indepVars.size());
    for (size_t i = 0; i < indepVars.size(); i++)
    {
        assert(indepVars[i].index < layer->data().n_cols);
        indepVarIndeces(i) = indepVars[i].index;
    }
    x = join_rows(mat(nDp, 1, arma::fill::ones), layer->data().cols(indepVarIndeces));
    y = layer->data().col(depVar.index);
}