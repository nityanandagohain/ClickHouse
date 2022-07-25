#pragma once

#include <Common/SharedLibrary.h>
#include <Common/logger_useful.h>

#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/IColumn.h>

#include "CatBoostLibraryAPI.h"

namespace DB
{

/// Abstracts access to CatBoost library.
class CatBoostLibraryHandler
{
    /// Holds pointers to CatBoost library functions
    struct APIHolder
    {
        CatBoostLibraryAPI::ModelCalcerCreateFunc ModelCalcerCreate; // NOLINT
        CatBoostLibraryAPI::ModelCalcerDeleteFunc ModelCalcerDelete; // NOLINT
        CatBoostLibraryAPI::GetErrorStringFunc GetErrorString; // NOLINT
        CatBoostLibraryAPI::LoadFullModelFromFileFunc LoadFullModelFromFile; // NOLINT
        CatBoostLibraryAPI::CalcModelPredictionFlatFunc CalcModelPredictionFlat; // NOLINT
        CatBoostLibraryAPI::CalcModelPredictionFunc CalcModelPrediction; // NOLINT
        CatBoostLibraryAPI::CalcModelPredictionWithHashedCatFeaturesFunc CalcModelPredictionWithHashedCatFeatures; // NOLINT
        CatBoostLibraryAPI::GetStringCatFeatureHashFunc GetStringCatFeatureHash; // NOLINT
        CatBoostLibraryAPI::GetIntegerCatFeatureHashFunc GetIntegerCatFeatureHash; // NOLINT
        CatBoostLibraryAPI::GetFloatFeaturesCountFunc GetFloatFeaturesCount; // NOLINT
        CatBoostLibraryAPI::GetCatFeaturesCountFunc GetCatFeaturesCount; // NOLINT
        CatBoostLibraryAPI::GetTreeCountFunc GetTreeCount; // NOLINT
        CatBoostLibraryAPI::GetDimensionsCountFunc GetDimensionsCount; // NOLINT

        void init(SharedLibrary & lib)
        {
            ModelCalcerCreate = lib.get<CatBoostLibraryAPI::ModelCalcerCreateFunc>(CatBoostLibraryAPI::ModelCalcerCreateName);
            ModelCalcerDelete = lib.get<CatBoostLibraryAPI::ModelCalcerDeleteFunc>(CatBoostLibraryAPI::ModelCalcerDeleteName);
            GetErrorString = lib.get<CatBoostLibraryAPI::GetErrorStringFunc>(CatBoostLibraryAPI::GetErrorStringName);
            LoadFullModelFromFile = lib.get<CatBoostLibraryAPI::LoadFullModelFromFileFunc>(CatBoostLibraryAPI::LoadFullModelFromFileName);
            CalcModelPredictionFlat = lib.get<CatBoostLibraryAPI::CalcModelPredictionFlatFunc>(CatBoostLibraryAPI::CalcModelPredictionFlatName);
            CalcModelPrediction = lib.get<CatBoostLibraryAPI::CalcModelPredictionFunc>(CatBoostLibraryAPI::CalcModelPredictionName);
            CalcModelPredictionWithHashedCatFeatures = lib.get<CatBoostLibraryAPI::CalcModelPredictionWithHashedCatFeaturesFunc>(CatBoostLibraryAPI::CalcModelPredictionWithHashedCatFeaturesName);
            GetStringCatFeatureHash = lib.get<CatBoostLibraryAPI::GetStringCatFeatureHashFunc>(CatBoostLibraryAPI::GetStringCatFeatureHashName);
            GetIntegerCatFeatureHash = lib.get<CatBoostLibraryAPI::GetIntegerCatFeatureHashFunc>(CatBoostLibraryAPI::GetIntegerCatFeatureHashName);
            GetFloatFeaturesCount = lib.get<CatBoostLibraryAPI::GetFloatFeaturesCountFunc>(CatBoostLibraryAPI::GetFloatFeaturesCountName);
            GetCatFeaturesCount = lib.get<CatBoostLibraryAPI::GetCatFeaturesCountFunc>(CatBoostLibraryAPI::GetCatFeaturesCountName);
            GetTreeCount = lib.tryGet<CatBoostLibraryAPI::GetTreeCountFunc>(CatBoostLibraryAPI::GetTreeCountName);
            GetDimensionsCount = lib.tryGet<CatBoostLibraryAPI::GetDimensionsCountFunc>(CatBoostLibraryAPI::GetDimensionsCountName);
        }
    };

public:
    CatBoostLibraryHandler(
        const std::string & library_path,
        const std::string & model_path);

    ~CatBoostLibraryHandler();

    ColumnPtr evaluate(const ColumnRawPtrs & columns) const;

private:
    SharedLibraryPtr library;
    APIHolder api;
    CatBoostLibraryAPI::ModelCalcerHandle * model_calcer_handle;

    size_t float_features_count;
    size_t cat_features_count;
    size_t tree_count;

    template <typename T>
    static void placeColumnAsNumber(const IColumn * column, T * buffer, size_t features_count);

    static void placeStringColumn(const ColumnString & column, const char ** buffer, size_t features_count);

    static PODArray<char> placeFixedStringColumn(const ColumnFixedString & column, const char ** buffer, size_t features_count);

    template <typename T>
    static ColumnPtr placeNumericColumns(const ColumnRawPtrs & columns, size_t offset, size_t size, const T** buffer);

    static std::vector<PODArray<char>> placeStringColumns(const ColumnRawPtrs & columns, size_t offset, size_t size, const char ** buffer);

    template <typename Column>
    static void calcStringHashes(const Column * column, size_t ps, const int ** buffer, const CatBoostLibraryHandler::APIHolder & api);

    static void calcIntHashes(size_t column_size, size_t ps, const int ** buffer, const CatBoostLibraryHandler::APIHolder & api);

    static void calcHashes(const ColumnRawPtrs & columns, size_t offset, size_t size, const int ** buffer, const CatBoostLibraryHandler::APIHolder & api);

    void fillCatFeaturesBuffer(const char *** cat_features, const char ** buffer, size_t column_size) const;

    ColumnFloat64::MutablePtr evalImpl(const ColumnRawPtrs & columns, bool cat_features_are_strings) const;
};

using CatBoostLibraryHandlerPtr = std::shared_ptr<CatBoostLibraryHandler>;

}
