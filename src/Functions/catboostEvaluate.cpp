#include <Functions/FunctionHelpers.h>
#include <Functions/FunctionFactory.h>

/// #include <base/range.h>
/// #include <Interpreters/Context.h>
/// #include <Interpreters/ExternalModelsLoader.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h> // temporary
/// #include <string>
/// #include <memory>
/// #include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypesNumber.h>
/// #include <Columns/ColumnNullable.h>
/// #include <Columns/ColumnTuple.h>
/// #include <DataTypes/DataTypeTuple.h>
/// #include <Common/assert_cast.h>
/// #include <Functions/IFunction.h>
#include <Interpreters/Context_fwd.h>

#include <BridgeHelper/LibraryBridgeHelper.h>
#include <BridgeHelper/CatBoostBridgeHelper.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int TOO_FEW_ARGUMENTS_FOR_FUNCTION;
    extern const int ILLEGAL_COLUMN;
}

class ExternalModelsLoader;


/// Evaluate CatBoost model.
/// - Arguments: float features first, then categorical features.
/// - Result: Float64.
class FunctionCatBoostEvaluate final : public IFunction, WithContext
{
public:
    static constexpr auto name = "catboostEvaluate";

    /// return std::make_shared<FunctionCatBoostEvaluate>(context->getExternalModelsLoader());
    static FunctionPtr create(ContextPtr context_) {return std::make_shared<FunctionCatBoostEvaluate>(context_);}

    explicit FunctionCatBoostEvaluate(ContextPtr context_) : WithContext(context_) {}
    String getName() const override { return name; }
    bool isVariadic() const override { return true; }
    bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return true; }
    bool isDeterministic() const override { return false; }
    bool useDefaultImplementationForNulls() const override { return false; }
    size_t getNumberOfArguments() const override { return 0; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        if (arguments.size() < 2)
            throw Exception("Function " + getName() + " expects at least 2 arguments",
                            ErrorCodes::TOO_FEW_ARGUMENTS_FOR_FUNCTION);

        if (!isString(arguments[0].type))
            throw Exception("Illegal type " + arguments[0].type->getName() + " of first argument of function " + getName()
                            + ", expected a string.", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

        const auto * name_col = checkAndGetColumnConst<ColumnString>(arguments[0].column.get());
        if (!name_col)
            throw Exception("First argument of function " + getName() + " must be a constant string",
                            ErrorCodes::ILLEGAL_COLUMN);

        bool has_nullable = false;
        for (size_t i = 1; i < arguments.size(); ++i)
            has_nullable = has_nullable || arguments[i].type->isNullable();

        /// TODO
        /// auto model = models_loader.getModel(name_col->getValue<String>());
        /// auto type = model->getReturnType();
        auto type = std::make_shared<DataTypeFloat64>();

        /// TODO
        /// if (has_nullable)
        /// {
        ///     if (const auto * tuple = typeid_cast<const DataTypeTuple *>(type.get()))
        ///     {
        ///         auto elements = tuple->getElements();
        ///         for (auto & element : elements)
        ///             element = makeNullable(element);
        ///
        ///         type = std::make_shared<DataTypeTuple>(elements);
        ///     }
        ///     else
        ///         type = makeNullable(type);
        /// }
        ///
        return type;
    }


    ColumnPtr executeImpl([[maybe_unused]] const ColumnsWithTypeAndName & arguments, [[maybe_unused]] const DataTypePtr &, size_t) const override // TODO
    {
        /// namespace fs = std::filesystem;
        ///
        /// auto dictionaries_lib_path = context->getDictionariesLibPath();
        /// if (created_from_ddl && !fileOrSymlinkPathStartsWith(path, dictionaries_lib_path))
        ///     throw Exception(ErrorCodes::PATH_ACCESS_DENIED, "File path {} is not inside {}", path, dictionaries_lib_path);
        ///
        /// if (!fs::exists(path))
        ///     throw Exception(ErrorCodes::FILE_DOESNT_EXIST, "LibraryDictionarySource: Can't load library {}: file doesn't exist", path);
        ///
        /// description.init(sample_block);
        ///
        CatBoostBridgeHelper::CatBoostInitData library_data
        {
            // TODO don't hardcode
            .library_path = "/home/ubuntu/catboost/data/libcatboostmodel.so",
            .model_path = "/home/ubuntu/catboost/models/amazon_model.xml"
            /// .library_path = path,
            /// .library_settings = getLibrarySettingsString(config, config_prefix + ".settings"),
            /// .dict_attributes = getDictAttributesString()
        };

        using CatBoostBridgeHelperPtr = std::shared_ptr<CatBoostBridgeHelper>;
        CatBoostBridgeHelperPtr bridge_helper = std::make_shared<CatBoostBridgeHelper>(getContext(), library_data);

        /// if (!bridge_helper->initLibrary())
        ///     throw Exception(ErrorCodes::EXTERNAL_LIBRARY_ERROR, "Failed to create shared library from path: {}", path);







        /// const auto * name_col = checkAndGetColumnConst<ColumnString>(arguments[0].column.get());
        /// if (!name_col)
        ///     throw Exception("First argument of function " + getName() + " must be a constant string",
        ///                     ErrorCodes::ILLEGAL_COLUMN);
        ///
        /// auto model = models_loader.getModel(name_col->getValue<String>());
        ///
        /// ColumnRawPtrs column_ptrs;
        /// Columns materialized_columns;
        /// ColumnPtr null_map;
        ///
        /// column_ptrs.reserve(arguments.size());
        /// for (auto arg : collections::range(1, arguments.size()))
        /// {
        ///     const auto & column = arguments[arg].column;
        ///     column_ptrs.push_back(column.get());
        ///     if (auto full_column = column->convertToFullColumnIfConst())
        ///     {
        ///         materialized_columns.push_back(full_column);
        ///         column_ptrs.back() = full_column.get();
        ///     }
        ///     if (const auto * col_nullable = checkAndGetColumn<ColumnNullable>(*column_ptrs.back()))
        ///     {
        ///         if (!null_map)
        ///             null_map = col_nullable->getNullMapColumnPtr();
        ///         else
        ///         {
        ///             auto mut_null_map = IColumn::mutate(std::move(null_map));
        ///
        ///             NullMap & result_null_map = assert_cast<ColumnUInt8 &>(*mut_null_map).getData();
        ///             const NullMap & src_null_map = col_nullable->getNullMapColumn().getData();
        ///
        ///             for (size_t i = 0, size = result_null_map.size(); i < size; ++i)
        ///                 if (src_null_map[i])
        ///                     result_null_map[i] = 1;
        ///
        ///             null_map = std::move(mut_null_map);
        ///         }
        ///
        ///         column_ptrs.back() = &col_nullable->getNestedColumn();
        ///     }
        /// }
        ///
        /// auto res = model->evaluate(column_ptrs);
        ///
        /// if (null_map)
        /// {
        ///     if (const auto * tuple = typeid_cast<const ColumnTuple *>(res.get()))
        ///     {
        ///         auto nested = tuple->getColumns();
        ///         for (auto & col : nested)
        ///             col = ColumnNullable::create(col, null_map);
        ///
        ///         res = ColumnTuple::create(nested);
        ///     }
        ///     else
        ///         res = ColumnNullable::create(res, null_map);
        /// }
        ///
        /// return res;

        /// return nullptr;
        auto res = ColumnFloat64::create(1); // temporary
        return res;
    }
};


REGISTER_FUNCTION(CatBoostEvaluate)
{
    factory.registerFunction<FunctionCatBoostEvaluate>();
}

}
