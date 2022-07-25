#include "SharedLibraryHandler.h"

#include <base/scope_guard.h>
#include <base/bit_cast.h>
#include <base/find_symbols.h>
#include <IO/ReadHelpers.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int EXTERNAL_LIBRARY_ERROR;
    extern const int SIZES_OF_COLUMNS_DOESNT_MATCH;
}

SharedLibraryHandler::SharedLibraryHandler(
    const std::string & library_path_,
    const std::vector<std::string> & library_settings,
    const Block & sample_block_,
    const std::vector<std::string> & attributes_names_)
    : library_path(library_path_)
    , sample_block(sample_block_)
    , attributes_names(attributes_names_)
{
    library = std::make_shared<SharedLibrary>(library_path);
    settings_holder = std::make_shared<CStringsHolder>(CStringsHolder(library_settings));

    auto lib_new = library->tryGet<SharedLibraryAPI::LibraryNewFunc>(SharedLibraryAPI::LIBRARY_CREATE_NEW_FUNC_NAME);

    if (lib_new)
        lib_data = lib_new(&settings_holder->strings, SharedLibraryAPI::log);
    else
        throw Exception("Method libNew failed", ErrorCodes::EXTERNAL_LIBRARY_ERROR);
}

SharedLibraryHandler::SharedLibraryHandler(const SharedLibraryHandler & other)
    : library_path{other.library_path}
    , sample_block{other.sample_block}
    , attributes_names{other.attributes_names}
    , library{other.library}
    , settings_holder{other.settings_holder}
{

    auto lib_clone = library->tryGet<SharedLibraryAPI::LibraryCloneFunc>(SharedLibraryAPI::LIBRARY_CLONE_FUNC_NAME);

    if (lib_clone)
    {
        lib_data = lib_clone(other.lib_data);
    }
    else
    {
        auto lib_new = library->tryGet<SharedLibraryAPI::LibraryNewFunc>(SharedLibraryAPI::LIBRARY_CREATE_NEW_FUNC_NAME);

        if (lib_new)
            lib_data = lib_new(&settings_holder->strings, SharedLibraryAPI::log);
    }
}

SharedLibraryHandler::~SharedLibraryHandler()
{
    auto lib_delete = library->tryGet<SharedLibraryAPI::LibraryDeleteFunc>(SharedLibraryAPI::LIBRARY_DELETE_FUNC_NAME);

    if (lib_delete)
        lib_delete(lib_data);
}

bool SharedLibraryHandler::isModified()
{
    auto func_is_modified = library->tryGet<SharedLibraryAPI::LibraryIsModifiedFunc>(SharedLibraryAPI::LIBRARY_IS_MODIFIED_FUNC_NAME);

    if (func_is_modified)
        return func_is_modified(lib_data, &settings_holder->strings);

    return true;
}

bool SharedLibraryHandler::supportsSelectiveLoad()
{
    auto func_supports_selective_load = library->tryGet<SharedLibraryAPI::LibrarySupportsSelectiveLoadFunc>(SharedLibraryAPI::LIBRARY_SUPPORTS_SELECTIVE_LOAD_FUNC_NAME);

    if (func_supports_selective_load)
        return func_supports_selective_load(lib_data, &settings_holder->strings);

    return true;
}

Block SharedLibraryHandler::loadAll()
{
    auto columns_holder = std::make_unique<SharedLibraryAPI::CString[]>(attributes_names.size());
    SharedLibraryAPI::CStrings columns{static_cast<decltype(SharedLibraryAPI::CStrings::data)>(columns_holder.get()), attributes_names.size()};
    for (size_t i = 0; i < attributes_names.size(); ++i)
        columns.data[i] = attributes_names[i].c_str();

    auto load_all_func = library->get<SharedLibraryAPI::LibraryLoadAllFunc>(SharedLibraryAPI::LIBRARY_LOAD_ALL_FUNC_NAME);
    auto data_new_func = library->get<SharedLibraryAPI::LibraryDataNewFunc>(SharedLibraryAPI::LIBRARY_DATA_NEW_FUNC_NAME);
    auto data_delete_func = library->get<SharedLibraryAPI::LibraryDataDeleteFunc>(SharedLibraryAPI::LIBRARY_DATA_DELETE_FUNC_NAME);

    SharedLibraryAPI::LibraryData data_ptr = data_new_func(lib_data);
    SCOPE_EXIT(data_delete_func(lib_data, data_ptr));

    SharedLibraryAPI::RawClickHouseLibraryTable data = load_all_func(data_ptr, &settings_holder->strings, &columns);
    return dataToBlock(data);
}

Block SharedLibraryHandler::loadIds(const std::vector<uint64_t> & ids)
{
    const SharedLibraryAPI::VectorUInt64 ids_data{bit_cast<decltype(SharedLibraryAPI::VectorUInt64::data)>(ids.data()), ids.size()};

    auto columns_holder = std::make_unique<SharedLibraryAPI::CString[]>(attributes_names.size());
    SharedLibraryAPI::CStrings columns_pass{static_cast<decltype(SharedLibraryAPI::CStrings::data)>(columns_holder.get()), attributes_names.size()};

    auto load_ids_func = library->get<SharedLibraryAPI::LibraryLoadIdsFunc>(SharedLibraryAPI::LIBRARY_LOAD_IDS_FUNC_NAME);
    auto data_new_func = library->get<SharedLibraryAPI::LibraryDataNewFunc>(SharedLibraryAPI::LIBRARY_DATA_NEW_FUNC_NAME);
    auto data_delete_func = library->get<SharedLibraryAPI::LibraryDataDeleteFunc>(SharedLibraryAPI::LIBRARY_DATA_DELETE_FUNC_NAME);

    SharedLibraryAPI::LibraryData data_ptr = data_new_func(lib_data);
    SCOPE_EXIT(data_delete_func(lib_data, data_ptr));

    SharedLibraryAPI::RawClickHouseLibraryTable data = load_ids_func(data_ptr, &settings_holder->strings, &columns_pass, &ids_data);
    return dataToBlock(data);
}

Block SharedLibraryHandler::loadKeys(const Columns & key_columns)
{
    auto holder = std::make_unique<SharedLibraryAPI::Row[]>(key_columns.size());
    std::vector<std::unique_ptr<SharedLibraryAPI::Field[]>> column_data_holders;

    for (size_t i = 0; i < key_columns.size(); ++i)
    {
        auto cell_holder = std::make_unique<SharedLibraryAPI::Field[]>(key_columns[i]->size());

        for (size_t j = 0; j < key_columns[i]->size(); ++j)
        {
            auto data_ref = key_columns[i]->getDataAt(j);

            cell_holder[j] = SharedLibraryAPI::Field{
                    .data = static_cast<const void *>(data_ref.data),
                    .size = data_ref.size};
        }

        holder[i] = SharedLibraryAPI::Row{
                    .data = static_cast<SharedLibraryAPI::Field *>(cell_holder.get()),
                    .size = key_columns[i]->size()};

        column_data_holders.push_back(std::move(cell_holder));
    }

    SharedLibraryAPI::Table request_cols{
            .data = static_cast<SharedLibraryAPI::Row *>(holder.get()),
            .size = key_columns.size()};

    auto load_keys_func = library->get<SharedLibraryAPI::LibraryLoadKeysFunc>(SharedLibraryAPI::LIBRARY_LOAD_KEYS_FUNC_NAME);
    auto data_new_func = library->get<SharedLibraryAPI::LibraryDataNewFunc>(SharedLibraryAPI::LIBRARY_DATA_NEW_FUNC_NAME);
    auto data_delete_func = library->get<SharedLibraryAPI::LibraryDataDeleteFunc>(SharedLibraryAPI::LIBRARY_DATA_DELETE_FUNC_NAME);

    SharedLibraryAPI::LibraryData data_ptr = data_new_func(lib_data);
    SCOPE_EXIT(data_delete_func(lib_data, data_ptr));

    SharedLibraryAPI::RawClickHouseLibraryTable data = load_keys_func(data_ptr, &settings_holder->strings, &request_cols);
    return dataToBlock(data);
}

Block SharedLibraryHandler::dataToBlock(SharedLibraryAPI::RawClickHouseLibraryTable data)
{
    if (!data)
        throw Exception("LibraryDictionarySource: No data returned", ErrorCodes::EXTERNAL_LIBRARY_ERROR);

    const auto * columns_received = static_cast<const SharedLibraryAPI::Table *>(data);
    if (columns_received->error_code)
        throw Exception(
            "LibraryDictionarySource: Returned error: " + std::to_string(columns_received->error_code) + " " + (columns_received->error_string ? columns_received->error_string : ""),
            ErrorCodes::EXTERNAL_LIBRARY_ERROR);

    MutableColumns columns = sample_block.cloneEmptyColumns();

    for (size_t col_n = 0; col_n < columns_received->size; ++col_n)
    {
        if (columns.size() != columns_received->data[col_n].size)
            throw Exception(
                "LibraryDictionarySource: Returned unexpected number of columns: " + std::to_string(columns_received->data[col_n].size) + ", must be " + std::to_string(columns.size()),
                ErrorCodes::SIZES_OF_COLUMNS_DOESNT_MATCH);

        for (size_t row_n = 0; row_n < columns_received->data[col_n].size; ++row_n)
        {
            const auto & field = columns_received->data[col_n].data[row_n];
            if (!field.data)
            {
                /// sample_block contains null_value (from config) inside corresponding column
                const auto & col = sample_block.getByPosition(row_n);
                columns[row_n]->insertFrom(*(col.column), 0);
            }
            else
            {
                const auto & size = field.size;
                columns[row_n]->insertData(static_cast<const char *>(field.data), size);
            }
        }
    }

    return sample_block.cloneWithColumns(std::move(columns));
}

}
