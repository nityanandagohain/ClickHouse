#pragma once

#include <Interpreters/Context.h>
#include <Server/HTTP/HTTPRequestHandler.h>
#include <Common/logger_useful.h>
#include "CatBoostLibraryHandler.h"

namespace DB
{

/// TODO docs
/// Handler for requests to Library Dictionary Source, returns response in RowBinary format.
/// When a library dictionary source is created, it sends libNew request to library bridge (which is started on first
/// request to it, if it was not yet started). On this request a new sharedLibrayHandler is added to a
/// sharedLibraryHandlerFactory by a dictionary uuid. With libNew request come: library_path, library_settings,
/// names of dictionary attributes, sample block to parse block of null values, block of null values. Everything is
/// passed in binary format and is urlencoded. When dictionary is cloned, a new handler is created.
/// Each handler is unique to dictionary.
class CatBoostBridgeRequestHandler : public HTTPRequestHandler, WithContext
{
public:

    CatBoostBridgeRequestHandler(size_t keep_alive_timeout_, ContextPtr context_);

    void handleRequest(HTTPServerRequest & request, HTTPServerResponse & response) override;

private:
    Poco::Logger * log;
    size_t keep_alive_timeout;
};


// Handler for checking if the CatBoost library is loaded (used for handshake)
class CatBoostBridgeLoadedHandler : public HTTPRequestHandler, WithContext
{
public:
    CatBoostBridgeLoadedHandler(size_t keep_alive_timeout_, ContextPtr context_);

    void handleRequest(HTTPServerRequest & request, HTTPServerResponse & response) override;

private:
    const size_t keep_alive_timeout;
    Poco::Logger * log;
};

}
