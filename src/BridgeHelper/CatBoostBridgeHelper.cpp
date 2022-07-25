#include "CatBoostBridgeHelper.h"

/// #include <Formats/formatBlock.h>
/// #include <Dictionaries/DictionarySourceHelpers.h>
/// #include <QueryPipeline/Pipe.h>
/// #include <Processors/Formats/IInputFormat.h>
/// #include <IO/WriteBufferFromOStream.h>
/// #include <IO/WriteBufferFromString.h>
/// #include <IO/ReadHelpers.h>
/// #include <IO/WriteHelpers.h>
/// #include <Formats/FormatFactory.h>
/// #include <Poco/Util/AbstractConfiguration.h>
/// #include <Common/ShellCommand.h>
/// #include <Common/logger_useful.h>
/// #include <base/range.h>
/// #include <Core/Field.h>
/// #include <Common/escapeForFileName.h>

#include <IO/ReadWriteBufferFromHTTP.h>
#include <Poco/Net/HTTPRequest.h>

namespace DB
{

namespace ErrorCodes
{
    /// extern const int EXTERNAL_LIBRARY_ERROR;
    /// extern const int LOGICAL_ERROR;
}

CatBoostBridgeHelper::CatBoostBridgeHelper(
        ContextPtr context_,
        const CatBoostInitData & catboost_data_)
    : IBridgeHelper(context_->getGlobalContext())
    , log(&Poco::Logger::get("CatBoostBridgeHelper"))
    , config(context_->getConfigRef())
    , http_timeout(context_->getGlobalContext()->getSettingsRef().http_receive_timeout.value)
    , catboost_data(catboost_data_)
    , bridge_host(config.getString("library_bridge.host", DEFAULT_HOST))
    , bridge_port(config.getUInt("library_bridge.port", DEFAULT_PORT))
    , http_timeouts(ConnectionTimeouts::getHTTPTimeouts(context_))
{
    startBridgeSync();
}


Poco::URI CatBoostBridgeHelper::createRequestURI(const String & method) const
{
    auto uri = getMainURI();
    /// uri.addQueryParameter("dictionary_id", toString(dictionary_id));
    uri.addQueryParameter("method", method);
    return uri;
}


Poco::URI CatBoostBridgeHelper::createBaseURI() const
{
    Poco::URI uri;
    uri.setHost(bridge_host);
    uri.setPort(bridge_port);
    uri.setScheme("http");
    return uri;
}


void CatBoostBridgeHelper::startBridge(std::unique_ptr<ShellCommand> cmd) const
{
    getContext()->addBridgeCommand(std::move(cmd));
}


bool CatBoostBridgeHelper::bridgeHandShake()
{
    String result;
    try
    {
        ReadWriteBufferFromHTTP buf(createRequestURI(PING), Poco::Net::HTTPRequest::HTTP_GET, {}, http_timeouts, credentials);
        readString(result, buf);
    }
    catch (...)
    {
        tryLogCurrentException(log);
        return false;
    }

    if (result.size() != 1)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Unexpected message from library bridge: {}. Check bridge and server have the same version.", result);

    /// UInt8 dictionary_id_exists;
    /// auto parsed = tryParse<UInt8>(dictionary_id_exists, result);
    /// if (!parsed || (dictionary_id_exists != 0 && dictionary_id_exists != 1))
    ///     throw Exception(ErrorCodes::LOGICAL_ERROR, "Unexpected message from library bridge: {} ({}). Check bridge and server have the same version.",
    ///                     result, parsed ? toString(dictionary_id_exists) : "failed to parse");
    ///
    /// LOG_TRACE(log, "dictionary_id: {}, dictionary_id_exists on bridge side: {}, library confirmed to be initialized on server side: {}",
    ///           toString(dictionary_id), toString(dictionary_id_exists), library_initialized);
    ///
    /// if (dictionary_id_exists && !library_initialized)
    ///     throw Exception(ErrorCodes::LOGICAL_ERROR, "Library was not initialized, but bridge responded to already have dictionary id: {}", dictionary_id);
    ///
    /// /// Here we want to say bridge to recreate a new library handler for current dictionary,
    /// /// because it responded to have lost it, but we know that it has already been created. (It is a direct result of bridge crash).
    /// if (!dictionary_id_exists && library_initialized)
    /// {
    ///     LOG_WARNING(log, "Library bridge does not have library handler with dictionaty id: {}. It will be reinitialized.", dictionary_id);
    ///     bool reinitialized = false;
    ///     try
    ///     {
    ///         auto uri = createRequestURI(LIB_NEW_METHOD);
    ///         reinitialized = executeRequest(uri, getInitLibraryCallback());
    ///     }
    ///     catch (...)
    ///     {
    ///         tryLogCurrentException(log);
    ///         return false;
    ///     }
    ///
    ///     if (!reinitialized)
    ///         throw Exception(ErrorCodes::EXTERNAL_LIBRARY_ERROR,
    ///                         "Failed to reinitialize library handler on bridge side for dictionary with id: {}", dictionary_id);
    /// }

    return true;
}


/// ReadWriteBufferFromHTTP::OutStreamCallback CatBoostBridgeHelper::getInitLibraryCallback() const
/// {
///     /// Sample block must contain null values
///     WriteBufferFromOwnString out;
///     auto output_format = getContext()->getOutputFormat(CatBoostBridgeHelper::DEFAULT_FORMAT, out, sample_block);
///     formatBlock(output_format, sample_block);
///     auto block_string = out.str();
///
///     return [block_string, this](std::ostream & os)
///     {
///         os << "library_path=" << escapeForFileName(catboost_data.library_path) << "&";
///         os << "library_settings=" << escapeForFileName(catboost_data.library_settings) << "&";
///         os << "attributes_names=" << escapeForFileName(catboost_data.dict_attributes) << "&";
///         os << "sample_block=" << escapeForFileName(sample_block.getNamesAndTypesList().toString()) << "&";
///         os << "null_values=" << escapeForFileName(block_string);
///     };
/// }
///
///
/// bool CatBoostBridgeHelper::initLibrary()
/// {
///     startBridgeSync();
///     auto uri = createRequestURI(LIB_NEW_METHOD);
///     library_initialized = executeRequest(uri, getInitLibraryCallback());
///     return library_initialized;
/// }
///
///
/// bool CatBoostBridgeHelper::cloneLibrary(const Field & other_dictionary_id)
/// {
///     startBridgeSync();
///     auto uri = createRequestURI(LIB_CLONE_METHOD);
///     uri.addQueryParameter("from_dictionary_id", toString(other_dictionary_id));
///     /// We also pass initialization settings in order to create a library handler
///     /// in case from_dictionary_id does not exist in bridge side (possible in case of bridge crash).
///     library_initialized = executeRequest(uri, getInitLibraryCallback());
///     return library_initialized;
/// }
///
///
/// bool CatBoostBridgeHelper::removeLibrary()
/// {
///     /// Do not force bridge restart if it is not running in case of removeLibrary
///     /// because in this case after restart it will not have this dictionaty id in memory anyway.
///     if (bridgeHandShake())
///     {
///         auto uri = createRequestURI(LIB_DELETE_METHOD);
///         return executeRequest(uri);
///     }
///     return true;
/// }
///
///
/// bool CatBoostBridgeHelper::isModified()
/// {
///     startBridgeSync();
///     auto uri = createRequestURI(IS_MODIFIED_METHOD);
///     return executeRequest(uri);
/// }
///
///
/// bool CatBoostBridgeHelper::supportsSelectiveLoad()
/// {
///     startBridgeSync();
///     auto uri = createRequestURI(SUPPORTS_SELECTIVE_LOAD_METHOD);
///     return executeRequest(uri);
/// }
///
///
/// QueryPipeline CatBoostBridgeHelper::loadAll()
/// {
///     startBridgeSync();
///     auto uri = createRequestURI(LOAD_ALL_METHOD);
///     return QueryPipeline(loadBase(uri));
/// }
///
///
/// QueryPipeline CatBoostBridgeHelper::loadIds(const std::vector<uint64_t> & ids)
/// {
///     startBridgeSync();
///     auto uri = createRequestURI(LOAD_IDS_METHOD);
///     uri.addQueryParameter("ids_num", toString(ids.size())); /// Not used parameter, but helpful
///     auto ids_string = getDictIdsString(ids);
///     return QueryPipeline(loadBase(uri, [ids_string](std::ostream & os) { os << ids_string; }));
/// }
///
///
/// QueryPipeline CatBoostBridgeHelper::loadKeys(const Block & requested_block)
/// {
///     startBridgeSync();
///     auto uri = createRequestURI(LOAD_KEYS_METHOD);
///     /// Sample block to parse block from callback
///     uri.addQueryParameter("requested_block_sample", requested_block.getNamesAndTypesList().toString());
///     ReadWriteBufferFromHTTP::OutStreamCallback out_stream_callback = [requested_block, this](std::ostream & os)
///     {
///         WriteBufferFromOStream out_buffer(os);
///         auto output_format = getContext()->getOutputFormat(CatBoostBridgeHelper::DEFAULT_FORMAT, out_buffer, requested_block.cloneEmpty());
///         formatBlock(output_format, requested_block);
///     };
///     return QueryPipeline(loadBase(uri, out_stream_callback));
/// }
///
///
/// bool CatBoostBridgeHelper::executeRequest(const Poco::URI & uri, ReadWriteBufferFromHTTP::OutStreamCallback out_stream_callback) const
/// {
///     ReadWriteBufferFromHTTP buf(
///         uri,
///         Poco::Net::HTTPRequest::HTTP_POST,
///         std::move(out_stream_callback),
///         http_timeouts, credentials);
///
///     bool res;
///     readBoolText(res, buf);
///     return res;
/// }
///
///
/// QueryPipeline CatBoostBridgeHelper::loadBase(const Poco::URI & uri, ReadWriteBufferFromHTTP::OutStreamCallback out_stream_callback)
/// {
///     auto read_buf_ptr = std::make_unique<ReadWriteBufferFromHTTP>(
///         uri,
///         Poco::Net::HTTPRequest::HTTP_POST,
///         std::move(out_stream_callback),
///         http_timeouts,
///         credentials,
///         0,
///         DBMS_DEFAULT_BUFFER_SIZE,
///         getContext()->getReadSettings(),
///         ReadWriteBufferFromHTTP::HTTPHeaderEntries{});
///
///     auto source = FormatFactory::instance().getInput(CatBoostBridgeHelper::DEFAULT_FORMAT, *read_buf_ptr, sample_block, getContext(), DEFAULT_BLOCK_SIZE);
///     source->addBuffer(std::move(read_buf_ptr));
///     return QueryPipeline(std::move(source));
/// }
///
///
/// String CatBoostBridgeHelper::getDictIdsString(const std::vector<UInt64> & ids)
/// {
///     WriteBufferFromOwnString out;
///     writeVectorBinary(ids, out);
///     return out.str();
/// }


}
