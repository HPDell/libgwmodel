#include "londonhp100.h"

bool read_londonhp100(mat& coords, mat& data, vector<string>& fields)
{
    field<std::string> coordHeader(2);
    coordHeader(0) = "x";
    coordHeader(1) = "y";
    bool coords_loaded = coords.load(arma::csv_name(string(SAMPLE_DATA_DIR) + "/londonhp100coords.csv", coordHeader));

    field<std::string> dataHeader(4);
    dataHeader(0) = "PURCHASE";
    dataHeader(1) = "FLOORSZ";
    dataHeader(2) = "UNEMPLOY";
    dataHeader(3) = "PROF";
    bool data_loaded = data.load(arma::csv_name(string(SAMPLE_DATA_DIR) + "/londonhp100data.csv", dataHeader));

    if (coords_loaded && data_loaded)
    {
        fields = { "PURCHASE", "FLOORSZ", "UNEMPLOY", "PROF" };
        return true;
    }
    else return false;
}