#include "CatBoostBridgeHandlerFactory.h"

#include <Poco/Net/HTTPServerRequest.h>
#include <Server/HTTP/HTMLForm.h>
#include "CatBoostBridgeHandlers.h"

namespace DB
{

CatBoostBridgeHandlerFactory::CatBoostBridgeHandlerFactory(
    const std::string & name_,
    size_t keep_alive_timeout_,
    ContextPtr context_)
    : WithContext(context_)
    , log(&Poco::Logger::get(name_))
    , name(name_)
    , keep_alive_timeout(keep_alive_timeout_)
{
}

std::unique_ptr<HTTPRequestHandler> CatBoostBridgeHandlerFactory::createRequestHandler(const HTTPServerRequest & request)
{
    Poco::URI uri{request.getURI()};
    LOG_DEBUG(log, "Request URI: {}", uri.toString());

    if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_GET)
        return std::make_unique<CatBoostBridgeLoadedHandler>(keep_alive_timeout, getContext());

    if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_POST)
        return std::make_unique<CatBoostBridgeRequestHandler>(keep_alive_timeout, getContext());

    return nullptr;
}

}
