#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include <gwmodel.h>
#include "londonhp100.h"

using namespace std;
using namespace arma;

vector<int> variables2indices(const GwmVariableListInterface& variables)
{
    vector<int> index;
    for (size_t i = 0; i < variables.size; i++)
    {
        index.push_back(variables.items[i].index);
    }
    return index;
}

mat interface2mat(const GwmMatInterface& interface)
{
    return mat(interface.data, interface.rows, interface.cols);
}

TEST_CASE("BasicGWR: Multithread")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    GwmMatInterface londonhp100_coord_interface; 
    londonhp100_coord_interface.rows = londonhp100_coord.n_rows;
    londonhp100_coord_interface.cols = londonhp100_coord.n_cols;
    londonhp100_coord_interface.data = londonhp100_coord.mem;
    GwmMatInterface londonhp100_data_interface;
    londonhp100_data_interface.rows = londonhp100_data.n_rows;
    londonhp100_data_interface.cols = londonhp100_data.n_cols;
    londonhp100_data_interface.data = londonhp100_data.mem;
    GwmNameListInterface londonhp100_field_interface;
    londonhp100_field_interface.size = londonhp100_fields.size();
    londonhp100_field_interface.items = new GwmNameInterface[londonhp100_field_interface.size];
    for (int i = 0; i < londonhp100_fields.size(); i++)
    {
        strcpy(londonhp100_field_interface.items[i], londonhp100_fields[i].data());
    }
    CGwmSimpleLayer* londonhp = gwmodel_create_simple_layer(londonhp100_coord_interface, londonhp100_data_interface, londonhp100_field_interface);

    CGwmDistance* distance = gwmodel_create_crs_distance(false);
    CGwmWeight* bandwidth = gwmodel_create_bandwidth_weight(36, true, KernelFunctionType::Gaussian);
    CGwmSpatialWeight* spatial = gwmodel_create_spatial_weight(distance, bandwidth);

    GwmVariableInterface purchase;
    purchase.index = 0;
    purchase.isNumeric = true;
    strcpy(purchase.name, "PURCHASE");
    GwmVariableInterface floorsz;
    floorsz.index = 1;
    floorsz.isNumeric = true;
    strcpy(floorsz.name, "FLOORSZ");
    GwmVariableInterface unemploy;
    unemploy.index = 2;
    unemploy.isNumeric = true;
    strcpy(unemploy.name, "UNEMPLOY");
    GwmVariableInterface prof;
    prof.index = 3;
    prof.isNumeric = true;
    strcpy(prof.name, "PROF");
    GwmVariableListInterface indepVars;
    indepVars.size = 3;
    indepVars.items = new GwmVariableInterface[indepVars.size];
    indepVars.items[0] = floorsz;
    indepVars.items[1] = unemploy;
    indepVars.items[2] = prof;

    CGwmGWRBasic* algorithm = gwmodel_create_gwr_algorithm();
    gwmodel_set_gwr_source_layer(algorithm, londonhp);
    gwmodel_set_gwr_dependent_variable(algorithm, purchase);
    gwmodel_set_gwr_independent_variable(algorithm, indepVars);
    gwmodel_set_gwr_spatial_weight(algorithm, spatial);
    gwmodel_set_gwr_options(algorithm, true);
    gwmodel_set_gwr_indep_vars_autoselection(algorithm, 3.0);
    gwmodel_set_gwr_bandwidth_autoselection(algorithm, BandwidthSelectionCriterionType::CV);
    gwmodel_set_gwr_openmp(algorithm, 8);
    REQUIRE_NOTHROW(gwmodel_run_gwr(algorithm));

    GwmVariablesCriterionListInterface criterionList = gwmodel_get_gwr_indep_var_criterions(algorithm);
    
    REQUIRE_THAT(variables2indices(criterionList.items[0].variables), Catch::Equals(vector<int>({ 2 })));
    REQUIRE(strcmp(criterionList.items[0].variables.items[0].name, "UNEMPLOY") == 0);
    REQUIRE_THAT(criterionList.items[0].criterion, Catch::WithinAbs(2551.61359020599, 1e-8));
    
    REQUIRE_THAT(variables2indices(criterionList.items[1].variables), Catch::Equals(vector<int>({ 3 })));
    REQUIRE_THAT(criterionList.items[1].criterion, Catch::WithinAbs(2551.30032201349, 1e-8));
    REQUIRE(strcmp(criterionList.items[1].variables.items[0].name, "PROF") == 0);

    REQUIRE_THAT(variables2indices(criterionList.items[2].variables), Catch::Equals(vector<int>({ 1 })));
    REQUIRE(strcmp(criterionList.items[2].variables.items[0].name, "FLOORSZ") == 0);
    REQUIRE_THAT(criterionList.items[2].criterion, Catch::WithinAbs(2468.93236280013, 1e-8));

    REQUIRE_THAT(variables2indices(criterionList.items[3].variables), Catch::Equals(vector<int>({ 1, 3 })));
    REQUIRE(strcmp(criterionList.items[3].variables.items[0].name, "FLOORSZ") == 0);
    REQUIRE(strcmp(criterionList.items[3].variables.items[1].name, "PROF") == 0);
    REQUIRE_THAT(criterionList.items[3].criterion, Catch::WithinAbs(2452.86447942033, 1e-8));

    REQUIRE_THAT(variables2indices(criterionList.items[4].variables), Catch::Equals(vector<int>({ 1, 2 })));
    REQUIRE(strcmp(criterionList.items[4].variables.items[0].name, "FLOORSZ") == 0);
    REQUIRE(strcmp(criterionList.items[4].variables.items[1].name, "UNEMPLOY") == 0);
    REQUIRE_THAT(criterionList.items[4].criterion, Catch::WithinAbs(2450.59642666509, 1e-8));

    REQUIRE_THAT(variables2indices(criterionList.items[5].variables), Catch::Equals(vector<int>({ 1, 2, 3 })));
    REQUIRE(strcmp(criterionList.items[5].variables.items[0].name, "FLOORSZ") == 0);
    REQUIRE(strcmp(criterionList.items[5].variables.items[1].name, "UNEMPLOY") == 0);
    REQUIRE(strcmp(criterionList.items[5].variables.items[2].name, "PROF") == 0);
    REQUIRE_THAT(criterionList.items[5].criterion, Catch::WithinAbs(2452.80388934625, 1e-8));

    CGwmSpatialWeight* spatial_new = gwmodel_get_gwr_spatial_weight(algorithm);

    GwmBandwidthKernelInterface bw_new_interf;
    if (gwmodel_get_spatial_bandwidth_weight(spatial_new, &bw_new_interf))
    {
        REQUIRE(bw_new_interf.size == 31);
    }
    else
    {
        FAIL("Cannot regard a CGwmWeight instance as a CGwmBandwidthWeight instance!");
    }

    GwmRegressionDiagnostic diagnostic = gwmodel_get_gwr_diagnostic(algorithm);
    REQUIRE_THAT(diagnostic.AIC, Catch::WithinAbs(2435.8161441795, 1e-8));
    REQUIRE_THAT(diagnostic.AICc, Catch::WithinAbs(2445.49629974057, 1e-8));
    REQUIRE_THAT(diagnostic.RSquare, Catch::WithinAbs(0.706143867720706, 1e-8));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::WithinAbs(0.678982114793865, 1e-8));

    REQUIRE_NOTHROW(gwmodel_delete_gwr_algorithm(algorithm));
    REQUIRE_NOTHROW(gwmodel_delete_string_list(&londonhp100_field_interface));
    REQUIRE_NOTHROW(gwmodel_delete_variable_list(&indepVars));
    REQUIRE_NOTHROW(gwmodel_delete_simple_layer(londonhp));
}

TEST_CASE("GWSS: Multithread")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    GwmMatInterface londonhp100_coord_interface; 
    londonhp100_coord_interface.rows = londonhp100_coord.n_rows;
    londonhp100_coord_interface.cols = londonhp100_coord.n_cols;
    londonhp100_coord_interface.data = londonhp100_coord.mem;
    GwmMatInterface londonhp100_data_interface;
    londonhp100_data_interface.rows = londonhp100_data.n_rows;
    londonhp100_data_interface.cols = londonhp100_data.n_cols;
    londonhp100_data_interface.data = londonhp100_data.mem;
    GwmNameListInterface londonhp100_field_interface;
    londonhp100_field_interface.size = londonhp100_fields.size();
    londonhp100_field_interface.items = new GwmNameInterface[londonhp100_field_interface.size];
    for (int i = 0; i < londonhp100_fields.size(); i++)
    {
        strcpy(londonhp100_field_interface.items[i], londonhp100_fields[i].data());
    }
    CGwmSimpleLayer* londonhp = gwmodel_create_simple_layer(londonhp100_coord_interface, londonhp100_data_interface, londonhp100_field_interface);

    CGwmDistance* distance = gwmodel_create_crs_distance(false);
    CGwmWeight* bandwidth = gwmodel_create_bandwidth_weight(36, true, KernelFunctionType::Gaussian);
    CGwmSpatialWeight* spatial = gwmodel_create_spatial_weight(distance, bandwidth);

    GwmVariableInterface purchase;
    purchase.index = 0;
    purchase.isNumeric = true;
    strcpy(purchase.name, "PURCHASE");
    GwmVariableInterface floorsz;
    floorsz.index = 1;
    floorsz.isNumeric = true;
    strcpy(floorsz.name, "FLOORSZ");
    GwmVariableInterface unemploy;
    unemploy.index = 2;
    unemploy.isNumeric = true;
    strcpy(unemploy.name, "UNEMPLOY");
    GwmVariableInterface prof;
    prof.index = 3;
    prof.isNumeric = true;
    strcpy(prof.name, "PROF");
    GwmVariableListInterface vars;
    vars.size = 4;
    vars.items = new GwmVariableInterface[vars.size];
    vars.items[0] = purchase;
    vars.items[1] = floorsz;
    vars.items[2] = unemploy;
    vars.items[3] = prof;

    CGwmGWSS* algorithm = gwmodel_create_gwss_algorithm();
    gwmodel_set_gwss_source_layer(algorithm, londonhp);
    gwmodel_set_gwss_variables(algorithm, vars);
    gwmodel_set_gwss_spatial_weight(algorithm, spatial);
    gwmodel_set_gwss_options(algorithm, false, false);
    gwmodel_set_gwss_openmp(algorithm, 8);
    REQUIRE_NOTHROW(gwmodel_run_gwss(algorithm));

    vec p = {0.0, 0.25, 0.5, 0.75, 1.0};

    mat localmean_q0 = {
        {155530.887621432,71.3459254279447,6.92671958853926,39.0446823327541},
        {163797.287583358,73.3206754261603,7.53173813461806,40.4678577700236},
        {174449.84375947,74.1174325820277,7.99672839037902,43.2051994175928},
        {183893.664229323,75.3118600659781,8.59668519607066,45.2164679493302},
        {188967.723827491,77.0911277060738,8.95571485750978,47.5614366837457}
    };
    GwmMatInterface local_mean = gwmodel_get_gwss_local_mean(algorithm);
    mat local_mean_mat = interface2mat(local_mean);
    mat localmean_q = quantile(local_mean_mat, p, 0);
    REQUIRE(approx_equal(localmean_q, localmean_q0, "absdiff", 1e-8));

    mat localsdev_q0 = {
        {72593.9921817404,28.3099770356131,2.01116607468286,8.4638277922896},
        {77170.1777015588,29.7600378924393,2.30743344312421,9.96621298558318},
        {80165.9773579845,30.1391577420805,2.38179570120204,10.7335766559347},
        {83051.0304538234,31.1706493250992,2.55775279583101,11.2114283718788},
        {86969.3221725472,32.2881606484993,2.73655762029611,12.0280808931404}
    };
    GwmMatInterface local_sdev = gwmodel_get_gwss_local_sdev(algorithm);
    mat localsdev_q = quantile(interface2mat(local_sdev), p, 0);
    REQUIRE(approx_equal(localsdev_q, localsdev_q0, "absdiff", 1e-8));

    mat localskew_q0 = {
        {1.4092924720023,1.37071524667066,-0.555779877884807,-0.111701647027156},
        {1.52855813553472,1.41818465286691,-0.306523936775326,0.0688874355072382},
        {1.7202801518175,1.48329930159006,0.00225908149445752,0.346270682414352},
        {2.01494028420899,1.65657212304686,0.278551178439127,0.529519631010928},
        {2.29647902712578,1.85491109594693,0.456786660462276,0.614689755724046}
    };
    GwmMatInterface local_skew = gwmodel_get_gwss_local_skew(algorithm);
    mat localskew_q = quantile(interface2mat(local_skew), p, 0);
    REQUIRE(approx_equal(localskew_q, localskew_q0, "absdiff", 1e-8));

    mat localcv_q0 = {
        {0.426450737915543,0.393305654809352,0.224567899568219,0.211056223726344},
        {0.437470421902242,0.405332689971102,0.272318025735197,0.230225532145268},
        {0.451648453541485,0.407743814376727,0.314108382503925,0.23750624244725},
        {0.490606242780692,0.411684812307897,0.334007325644591,0.26722230789406},
        {0.520406719170266,0.419294146622537,0.339426249173607,0.2901354343712}
    };
    GwmMatInterface local_cv = gwmodel_get_gwss_local_cv(algorithm);
    mat localcv_q = quantile(interface2mat(local_cv), p, 0);
    REQUIRE(approx_equal(localcv_q, localcv_q0, "absdiff", 1e-8));

    mat localcorr_q0 = {
        {0.748948486801849,-0.320600183598632,0.203011140141453,-0.126882976445561,-0.0892568204410789,-0.948799446008617},
        {0.762547101624896,-0.297490396583388,0.246560726908457,-0.0855566960390598,-0.0108045673038358,-0.939669443787772},
        {0.78483823103956,-0.254451851221453,0.282830629241902,-0.0466716717586483,0.0860667834905564,-0.930939912151655},
        {0.809708575169509,-0.240880285636795,0.324300091997221,0.030563100930792,0.14224438342357,-0.924880287887928},
        {0.838005736892351,-0.201907496636598,0.35265748682446,0.106759558870671,0.161751404622356,-0.906594939821811}
    };
    GwmMatInterface local_corr = gwmodel_get_gwss_local_corr(algorithm);
    mat localcorr_q = quantile(interface2mat(local_corr), p, 0);
    REQUIRE(approx_equal(localcorr_q, localcorr_q0, "absdiff", 1e-8));

    mat localscorr_q0 = {
        {0.521222457142438,-0.386537315399977,0.272098100316185,-0.132913057346789,-0.0706904961467669,-0.940629495956178},
        {0.546058956484106,-0.367722715928213,0.28224553968716,-0.100607936868221,0.00362924865128611,-0.931506008150178},
        {0.591076906824072,-0.333869710084257,0.336014460751443,-0.0756778419096376,0.0739387878352967,-0.928259365660612},
        {0.642395389246104,-0.314342558536871,0.358117991041394,-0.000170454849373912,0.108476590000141,-0.915878602985333},
        {0.685066744419873,-0.296544286394518,0.380785226148097,0.0690739762835091,0.170298974146835,-0.895252623185884}
    };
    GwmMatInterface local_spearman_rho = gwmodel_get_gwss_local_spearman_rho(algorithm);
    mat localscorr_q = quantile(interface2mat(local_spearman_rho), p, 0);
    REQUIRE(approx_equal(localscorr_q, localscorr_q0, "absdiff", 1e-1));

    REQUIRE_NOTHROW(gwmodel_delete_mat(&local_mean));
    REQUIRE_NOTHROW(gwmodel_delete_mat(&local_sdev));
    REQUIRE_NOTHROW(gwmodel_delete_mat(&local_skew));
    REQUIRE_NOTHROW(gwmodel_delete_mat(&local_cv));
    REQUIRE_NOTHROW(gwmodel_delete_mat(&local_corr));
    REQUIRE_NOTHROW(gwmodel_delete_mat(&local_spearman_rho));
    REQUIRE_NOTHROW(gwmodel_delete_gwss_algorithm(algorithm));
    REQUIRE_NOTHROW(gwmodel_delete_string_list(&londonhp100_field_interface));
    REQUIRE_NOTHROW(gwmodel_delete_variable_list(&vars));
    REQUIRE_NOTHROW(gwmodel_delete_simple_layer(londonhp));
}