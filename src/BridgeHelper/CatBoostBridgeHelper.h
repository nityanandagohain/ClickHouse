#pragma once

#include <Interpreters/Context.h>
#include <Poco/Logger.h>
#include <Poco/URI.h>
#include <BridgeHelper/IBridgeHelper.h>
#include <Poco/Net/HTTPBasicCredentials.h>
/// #include <QueryPipeline/QueryPipeline.h>


namespace DB
{

/// class Pipe;

class CatBoostBridgeHelper : public IBridgeHelper
{

public:
    struct CatBoostInitData
    {
        String library_path;
        String model_path;
    };

    static constexpr inline size_t DEFAULT_PORT = 9013;

    CatBoostBridgeHelper(ContextPtr context_, const CatBoostInitData & library_data_);

    /// bool initLibrary();
    ///
    /// bool cloneLibrary(const Field & other_dictionary_id);
    ///
    /// bool removeLibrary();
    ///
    /// bool isModified();
    ///
    /// bool supportsSelectiveLoad();
    ///
    /// QueryPipeline loadAll();
    ///
    /// QueryPipeline loadIds(const std::vector<uint64_t> & ids);
    ///
    /// QueryPipeline loadKeys(const Block & requested_block);
    ///
    /// QueryPipeline loadBase(const Poco::URI & uri, ReadWriteBufferFromHTTP::OutStreamCallback out_stream_callback = {});
    ///
    /// bool executeRequest(const Poco::URI & uri, ReadWriteBufferFromHTTP::OutStreamCallback out_stream_callback = {}) const;
    ///
    /// CatBoostInitData getLibraryData() const { return catboost_data; }

protected:
    bool bridgeHandShake() override;

    void startBridge(std::unique_ptr<ShellCommand> cmd) const override;

    String serviceAlias() const override { return "clickhouse-catboost-bridge"; }

    String serviceFileName() const override { return serviceAlias(); }

    size_t getDefaultPort() const override { return DEFAULT_PORT; }

    bool startBridgeManually() const override { return false; }

    String configPrefix() const override { return "library_bridge"; }

    const Poco::Util::AbstractConfiguration & getConfig() const override { return config; }

    Poco::Logger * getLog() const override { return log; }

    Poco::Timespan getHTTPTimeout() const override { return http_timeout; }

    Poco::URI createBaseURI() const override;

    /// ReadWriteBufferFromHTTP::OutStreamCallback getInitLibraryCallback() const;

private:
    static constexpr inline auto LIB_NEW_METHOD = "libNew";
    /// static constexpr inline auto LIB_CLONE_METHOD = "libClone";
    /// static constexpr inline auto LIB_DELETE_METHOD = "libDelete";
    /// static constexpr inline auto LOAD_ALL_METHOD = "loadAll";
    /// static constexpr inline auto LOAD_IDS_METHOD = "loadIds";
    /// static constexpr inline auto LOAD_KEYS_METHOD = "loadKeys";
    /// static constexpr inline auto IS_MODIFIED_METHOD = "isModified";
    static constexpr inline auto PING = "ping";
    /// static constexpr inline auto SUPPORTS_SELECTIVE_LOAD_METHOD = "supportsSelectiveLoad";

    Poco::URI createRequestURI(const String & method) const;

    /// static String getDictIdsString(const std::vector<UInt64> & ids);
    ///
    Poco::Logger * log;
    const Poco::Util::AbstractConfiguration & config;
    const Poco::Timespan http_timeout;

    CatBoostInitData catboost_data;
    std::string bridge_host;
    size_t bridge_port;
    /// bool library_initialized = false;
    ConnectionTimeouts http_timeouts;
    Poco::Net::HTTPBasicCredentials credentials{};
};

}
