#pragma once

#include "CatBoostLibraryHandler.h"
#include <base/defines.h>

#include <mutex>


namespace DB
{

class CatBoostLibraryHandlerFactory final : private boost::noncopyable
{
public:
    static CatBoostLibraryHandlerFactory & instance();

    // TODO needed? (maybe we don't need the calling handler)
    CatBoostLibraryHandlerPtr get();

    void create(const std::string & library_path, const std::string & model_path);

    bool reset();

private:
    CatBoostLibraryHandlerPtr library_handler TSA_GUARDED_BY(mutex);
    std::mutex mutex;
};

}
