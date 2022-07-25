#include "CatBoostBridgeHandlers.h"
#include "CatBoostLibraryHandlerFactory.h"

#include <Formats/FormatFactory.h>
#include <Server/HTTP/WriteBufferFromHTTPServerResponse.h>
#include <IO/WriteHelpers.h>
#include <IO/ReadHelpers.h>
#include <Poco/Net/HTTPServerRequest.h>
#include <Poco/Net/HTTPServerResponse.h>
#include <Poco/Net/HTMLForm.h>
#include <Poco/ThreadPool.h>
#include <Processors/Formats/IOutputFormat.h>
#include <Processors/Formats/IInputFormat.h>
#include <QueryPipeline/QueryPipeline.h>
#include <Processors/Executors/CompletedPipelineExecutor.h>
#include <Processors/Executors/PullingPipelineExecutor.h>
#include <Processors/Sources/SourceFromSingleChunk.h>
#include <QueryPipeline/Pipe.h>
#include <Server/HTTP/HTMLForm.h>
#include <IO/ReadBufferFromString.h>

// TODO clean up includes in all files of catboost-bridge
// TODO same struct structure also for library-bridge
// TODO: prefix log message with file name

namespace DB
{

namespace
{

void processError(HTTPServerResponse & response, const std::string & message)
{
    response.setStatusAndReason(HTTPResponse::HTTP_INTERNAL_SERVER_ERROR);

    if (!response.sent())
        *response.send() << message << std::endl;

    LOG_WARNING(&Poco::Logger::get("CatBoostLibraryBridge"), fmt::runtime(message));
}
}

CatBoostBridgeRequestHandler::CatBoostBridgeRequestHandler(
    size_t keep_alive_timeout_, ContextPtr context_)
    : WithContext(context_)
    , log(&Poco::Logger::get("CatBoostBridgeRequestHandler"))
    , keep_alive_timeout(keep_alive_timeout_)
{
}

void CatBoostBridgeRequestHandler::handleRequest(HTTPServerRequest & request, HTTPServerResponse & response)
{
    LOG_TRACE(log, "Request URI: {}", request.getURI());
    HTMLForm params(getContext()->getSettingsRef(), request);

    if (!params.has("method"))
    {
        processError(response, "No 'method' in request URL");
        return;
    }

    std::string method = params.get("method");

    LOG_TRACE(log, "Library method: '{}'", method);
    WriteBufferFromHTTPServerResponse out(response, request.getMethod() == Poco::Net::HTTPRequest::HTTP_HEAD, keep_alive_timeout);

    try
    {
        if (method == "libNew")
        {
            auto & read_buf = request.getStream();
            params.read(read_buf);

            if (!params.has("library_path"))
            {
                processError(response, "No 'library_path' in request URL");
                return;
            }

            std::string library_path = params.get("library_path");

            if (!params.has("model_path"))
            {
                processError(response, "No 'model_path' in request URL");
                return;
            }

            std::string model_path = params.get("model_path");

            CatBoostLibraryHandlerFactory::instance().create(library_path, model_path);
            writeStringBinary("1", out);
        }
        else if (method == "libDelete")
        {
            bool deleted = CatBoostLibraryHandlerFactory::instance().reset();

            /// Do not throw, a warning is ok.
            if (!deleted)
                LOG_WARNING(log, "Did not delete catboost library because it was not loaded.");

            writeStringBinary("1", out);
        }
        else if (false)
        {
            // TODO more stuff
        }
        else
        {
            LOG_WARNING(log, "Unknown library method: '{}'", method);
        }
    }
    catch (...)
    {
        auto message = getCurrentExceptionMessage(true);
        LOG_ERROR(log, "Failed to process request. Error: {}", message);

        response.setStatusAndReason(Poco::Net::HTTPResponse::HTTP_INTERNAL_SERVER_ERROR, message); // can't call process_error, because of too soon response sending
        try
        {
            writeStringBinary(message, out);
            out.finalize();
        }
        catch (...)
        {
            tryLogCurrentException(log);
        }
    }

    try
    {
        out.finalize();
    }
    catch (...)
    {
        tryLogCurrentException(log);
    }
}

CatBoostBridgeLoadedHandler::CatBoostBridgeLoadedHandler(size_t keep_alive_timeout_, ContextPtr context_)
    : WithContext(context_)
    , keep_alive_timeout(keep_alive_timeout_)
    , log(&Poco::Logger::get("CatBoostBridgeLoadedHandler"))
{
}

void CatBoostBridgeLoadedHandler::handleRequest(HTTPServerRequest & request, [[maybe_unused]] HTTPServerResponse & response)
{
    try
    {
        LOG_TRACE(log, "Request URI: {}", request.getURI());
        HTMLForm params(getContext()->getSettingsRef(), request);

        auto catboost_handler = CatBoostLibraryHandlerFactory::instance().get();

        String res = catboost_handler ? "1" : "0";
        setResponseDefaultHeaders(response, keep_alive_timeout);
        LOG_TRACE(log, "Sending ping response: {}", res);
        response.sendBuffer(res.data(), res.size());
    }
    catch (...)
    {
        tryLogCurrentException("PingHandler");
    }
}

}
