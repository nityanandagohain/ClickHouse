#include "CatBoostLibraryHandler.h"

#include <Columns/ColumnTuple.h>
#include <Common/FieldVisitorConvertToNumber.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int CANNOT_APPLY_CATBOOST_MODEL;
    extern const int CANNOT_LOAD_CATBOOST_MODEL;
}

// TODO error handling, e.g. library path does not exist, model path does not exist etc.
// TODO apply stuff done to library-bridge also to odbc bridge

CatBoostLibraryHandler::CatBoostLibraryHandler(
    const std::string & library_path,
    const std::string & model_path)
{
    library = std::make_shared<SharedLibrary>(library_path);
    api.init(*library);

    model_calcer_handle = api.ModelCalcerCreate();

    if (!api.LoadFullModelFromFile(model_calcer_handle, model_path.c_str()))
    {
        throw Exception(ErrorCodes::CANNOT_LOAD_CATBOOST_MODEL,
                "Cannot load CatBoost model: {}", api.GetErrorString());
    }

    float_features_count = api.GetFloatFeaturesCount(model_calcer_handle);
    cat_features_count = api.GetCatFeaturesCount(model_calcer_handle);

    tree_count = 1;
    if (api.GetDimensionsCount)
        tree_count = api.GetDimensionsCount(model_calcer_handle);
}

CatBoostLibraryHandler::~CatBoostLibraryHandler()
{
    api.ModelCalcerDelete(model_calcer_handle);
}

/// Buffer should be allocated with features_count * column->size() elements.
/// Place column elements in positions buffer[0], buffer[features_count], ... , buffer[size * features_count]
template <typename T>
void CatBoostLibraryHandler::placeColumnAsNumber(
    const IColumn * column, T * buffer, size_t features_count)
{
    size_t size = column->size();
    FieldVisitorConvertToNumber<T> visitor;
    for (size_t i = 0; i < size; ++i)
    {
        /// TODO: Replace with column visitor.
        Field field;
        column->get(i, field);
        *buffer = applyVisitor(visitor, field);
        buffer += features_count;
    }
}

/// Buffer should be allocated with features_count * column->size() elements.
/// Place string pointers in positions buffer[0], buffer[features_count], ... , buffer[size * features_count]
void CatBoostLibraryHandler::placeStringColumn(
    const ColumnString & column, const char ** buffer, size_t features_count)
{
    size_t size = column.size();
    for (size_t i = 0; i < size; ++i)
    {
        *buffer = const_cast<char *>(column.getDataAtWithTerminatingZero(i).data);
        buffer += features_count;
    }
}

/// Buffer should be allocated with features_count * column->size() elements.
/// Place string pointers in positions buffer[0], buffer[features_count], ... , buffer[size * features_count]
/// Returns PODArray which holds data (because ColumnFixedString doesn't store terminating zero).
PODArray<char> CatBoostLibraryHandler::placeFixedStringColumn(
    const ColumnFixedString & column, const char ** buffer, size_t features_count)
{
    size_t size = column.size();
    size_t str_size = column.getN();
    PODArray<char> data(size * (str_size + 1));
    char * data_ptr = data.data();

    for (size_t i = 0; i < size; ++i)
    {
        auto ref = column.getDataAt(i);
        memcpy(data_ptr, ref.data, ref.size);
        data_ptr[ref.size] = 0;
        *buffer = data_ptr;
        data_ptr += ref.size + 1;
        buffer += features_count;
    }

    return data;
}

/// Place columns into buffer, returns column which holds placed data. Buffer should contains column->size() values.
template <typename T>
ColumnPtr CatBoostLibraryHandler::placeNumericColumns(
    const ColumnRawPtrs & columns,
    size_t offset, size_t size, const T** buffer)
{
    if (size == 0)
        return nullptr;

    size_t column_size = columns[offset]->size();
    auto data_column = ColumnVector<T>::create(size * column_size);
    T * data = data_column->getData().data();
    for (size_t i = 0; i < size; ++i)
    {
        const auto * column = columns[offset + i];
        if (column->isNumeric())
            placeColumnAsNumber(column, data + i, size);
    }

    for (size_t i = 0; i < column_size; ++i)
    {
        *buffer = data;
        ++buffer;
        data += size;
    }

    return data_column;
}

/// Place columns into buffer, returns data which was used for fixed string columns.
/// Buffer should contains column->size() values, each value contains size strings.
std::vector<PODArray<char>> CatBoostLibraryHandler::placeStringColumns(
    const ColumnRawPtrs & columns,
    size_t offset, size_t size, const char ** buffer)
{
    if (size == 0)
        return {};

    std::vector<PODArray<char>> data;
    for (size_t i = 0; i < size; ++i)
    {
        const auto * column = columns[offset + i];
        if (const auto * column_string = typeid_cast<const ColumnString *>(column))
            placeStringColumn(*column_string, buffer + i, size);
        else if (const auto * column_fixed_string = typeid_cast<const ColumnFixedString *>(column))
            data.push_back(placeFixedStringColumn(*column_fixed_string, buffer + i, size));
        else
            throw Exception("Cannot place string column.", ErrorCodes::LOGICAL_ERROR);
    }

    return data;
}

/// Calc hash for string cat feature at ps positions.
template <typename Column>
void CatBoostLibraryHandler::calcStringHashes(
    const Column * column, size_t ps, const int ** buffer, const CatBoostLibraryHandler::APIHolder & api)
{
    size_t column_size = column->size();
    for (size_t j = 0; j < column_size; ++j)
    {
        auto ref = column->getDataAt(j);
        const_cast<int *>(*buffer)[ps] = api.GetStringCatFeatureHash(ref.data, ref.size);
        ++buffer;
    }
}

/// Calc hash for int cat feature at ps position. Buffer at positions ps should contains unhashed values.
void CatBoostLibraryHandler::calcIntHashes(
    size_t column_size, size_t ps, const int ** buffer, const CatBoostLibraryHandler::APIHolder & api)
{
    for (size_t j = 0; j < column_size; ++j)
    {
        const_cast<int *>(*buffer)[ps] = api.GetIntegerCatFeatureHash((*buffer)[ps]);
        ++buffer;
    }
}

/// buffer contains column->size() rows and size columns.
/// For int cat features calc hash inplace.
/// For string cat features calc hash from column rows.
void CatBoostLibraryHandler::calcHashes(
    const ColumnRawPtrs & columns, size_t offset, size_t size, const int ** buffer, const CatBoostLibraryHandler::APIHolder & api)
{
    if (size == 0)
        return;
    size_t column_size = columns[offset]->size();

    std::vector<PODArray<char>> data;
    for (size_t i = 0; i < size; ++i)
    {
        const auto * column = columns[offset + i];
        if (const auto * column_string = typeid_cast<const ColumnString *>(column))
            calcStringHashes(column_string, i, buffer, api);
        else if (const auto * column_fixed_string = typeid_cast<const ColumnFixedString *>(column))
            calcStringHashes(column_fixed_string, i, buffer, api);
        else
            calcIntHashes(column_size, i, buffer, api);
    }
}

/// buffer[column_size * cat_features_count] -> char * => cat_features[column_size][cat_features_count] -> char *
void CatBoostLibraryHandler::fillCatFeaturesBuffer(
    const char *** cat_features, const char ** buffer,
    size_t column_size) const
{
    for (size_t i = 0; i < column_size; ++i)
    {
        *cat_features = buffer;
        ++cat_features;
        buffer += cat_features_count;
    }
}

/// Convert values to row-oriented format and call evaluation function from CatBoost wrapper api.
///  * CalcModelPredictionFlat if no cat features
///  * CalcModelPrediction if all cat features are strings
///  * CalcModelPredictionWithHashedCatFeatures if has int cat features.
ColumnFloat64::MutablePtr CatBoostLibraryHandler::evalImpl(
    const ColumnRawPtrs & columns,
    bool cat_features_are_strings) const
{
    std::string error_msg = "Error occurred while applying CatBoost model: ";
    size_t column_size = columns.front()->size();

    auto result = ColumnFloat64::create(column_size * tree_count);
    auto * result_buf = result->getData().data();

    if (!column_size)
        return result;

    /// Prepare float features.
    PODArray<const float *> float_features(column_size);
    auto * float_features_buf = float_features.data();
    /// Store all float data into single column. float_features is a list of pointers to it.
    auto float_features_col = placeNumericColumns<float>(columns, 0, float_features_count, float_features_buf);

    if (cat_features_count == 0)
    {
        if (!api.CalcModelPredictionFlat(model_calcer_handle, column_size,
                                          float_features_buf, float_features_count,
                                          result_buf, column_size * tree_count))
        {

            throw Exception(error_msg + api.GetErrorString(), ErrorCodes::CANNOT_APPLY_CATBOOST_MODEL);
        }
        return result;
    }

    /// Prepare cat features.
    if (cat_features_are_strings)
    {
        /// cat_features_holder stores pointers to ColumnString data or fixed_strings_data.
        PODArray<const char *> cat_features_holder(cat_features_count * column_size);
        PODArray<const char **> cat_features(column_size);
        auto * cat_features_buf = cat_features.data();

        fillCatFeaturesBuffer(cat_features_buf, cat_features_holder.data(), column_size);
        /// Fixed strings are stored without termination zero, so have to copy data into fixed_strings_data.
        auto fixed_strings_data = placeStringColumns(columns, float_features_count,
                                                     cat_features_count, cat_features_holder.data());

        if (!api.CalcModelPrediction(model_calcer_handle, column_size,
                                      float_features_buf, float_features_count,
                                      cat_features_buf, cat_features_count,
                                      result_buf, column_size * tree_count))
        {
            throw Exception(error_msg + api.GetErrorString(), ErrorCodes::CANNOT_APPLY_CATBOOST_MODEL);
        }
    }
    else
    {
        PODArray<const int *> cat_features(column_size);
        auto * cat_features_buf = cat_features.data();
        auto cat_features_col = placeNumericColumns<int>(columns, float_features_count,
                                                         cat_features_count, cat_features_buf);
        calcHashes(columns, float_features_count, cat_features_count, cat_features_buf, api);
        if (!api.CalcModelPredictionWithHashedCatFeatures(
                model_calcer_handle, column_size,
                float_features_buf, float_features_count,
                cat_features_buf, cat_features_count,
                result_buf, column_size * tree_count))
        {
            throw Exception(error_msg + api.GetErrorString(), ErrorCodes::CANNOT_APPLY_CATBOOST_MODEL);
        }
    }

    return result;
}

// TODO needs serialization/deserialization wrapper
// TODO check that the library is initialized

ColumnPtr CatBoostLibraryHandler::evaluate(const ColumnRawPtrs & columns) const
{
    if (columns.empty())
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Got empty columns list for CatBoost model.");

    if (columns.size() != float_features_count + cat_features_count)
        throw Exception(ErrorCodes::BAD_ARGUMENTS,
            "Number of columns is different with number of features: columns size {} float features size {} + cat features size {}",
            columns.size(),
            float_features_count,
            cat_features_count);

    for (size_t i = 0; i < float_features_count; ++i)
    {
        if (!columns[i]->isNumeric())
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Column {} should be numeric to make float feature.", i);
        }
    }

    bool cat_features_are_strings = true;
    for (size_t i = float_features_count; i < float_features_count + cat_features_count; ++i)
    {
        const auto * column = columns[i];
        if (column->isNumeric())
        {
            cat_features_are_strings = false;
        }
        else if (!(typeid_cast<const ColumnString *>(column)
                   || typeid_cast<const ColumnFixedString *>(column)))
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Column {} should be numeric or string.", i);
        }
    }

    auto result = evalImpl(columns, cat_features_are_strings);

    if (tree_count == 1)
        return result;

    size_t column_size = columns.front()->size();
    auto * result_buf = result->getData().data();

    /// Multiple trees case. Copy data to several columns.
    MutableColumns mutable_columns(tree_count);
    std::vector<Float64 *> column_ptrs(tree_count);
    for (size_t i = 0; i < tree_count; ++i)
    {
        auto col = ColumnFloat64::create(column_size);
        column_ptrs[i] = col->getData().data();
        mutable_columns[i] = std::move(col);
    }

    Float64 * data = result_buf;
    for (size_t row = 0; row < column_size; ++row)
    {
        for (size_t i = 0; i < tree_count; ++i)
        {
            *column_ptrs[i] = *data;
            ++column_ptrs[i];
            ++data;
        }
    }

    return ColumnTuple::create(std::move(mutable_columns));
}

}
