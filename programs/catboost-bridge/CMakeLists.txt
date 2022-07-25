include(${ClickHouse_SOURCE_DIR}/cmake/split_debug_symbols.cmake)

set (CLICKHOUSE_CATBOOST_BRIDGE_SOURCES
    CatBoostBridge.cpp
    CatBoostBridgeHandlerFactory.cpp
    CatBoostBridgeHandlers.cpp
    CatBoostLibraryHandler.cpp
    CatBoostLibraryHandlerFactory.cpp
)

if (OS_LINUX)
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-export-dynamic")
endif ()

clickhouse_add_executable(clickhouse-catboost-bridge ${CLICKHOUSE_CATBOOST_BRIDGE_SOURCES})

target_link_libraries(clickhouse-catboost-bridge PRIVATE
    daemon
    dbms
    bridge
)

set_target_properties(clickhouse-catboost-bridge PROPERTIES RUNTIME_OUTPUT_DIRECTORY ..)

if (SPLIT_DEBUG_SYMBOLS)
    clickhouse_split_debug_symbols(TARGET clickhouse-catboost-bridge DESTINATION_DIR ${CMAKE_CURRENT_BINARY_DIR}/../${SPLITTED_DEBUG_SYMBOLS_DIR} BINARY_PATH ../clickhouse-catboost-bridge)
else()
    clickhouse_make_empty_debug_info_for_nfpm(TARGET clickhouse-catboost-bridge DESTINATION_DIR ${CMAKE_CURRENT_BINARY_DIR}/../${SPLITTED_DEBUG_SYMBOLS_DIR})
    install(TARGETS clickhouse-catboost-bridge RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT clickhouse)
endif()
