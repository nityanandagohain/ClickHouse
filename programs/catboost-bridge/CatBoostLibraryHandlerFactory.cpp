#include "CatBoostLibraryHandlerFactory.h"

namespace DB
{

CatBoostLibraryHandlerFactory & CatBoostLibraryHandlerFactory::instance()
{
    static CatBoostLibraryHandlerFactory instance;
    return instance;
}

CatBoostLibraryHandlerPtr CatBoostLibraryHandlerFactory::get()
{
    std::lock_guard lock(mutex);
    return library_handler;
}

void CatBoostLibraryHandlerFactory::create(const std::string & library_path, const std::string & model_path)
{
    std::lock_guard lock(mutex);
    if (!library_handler)
    {
        library_handler = std::make_shared<CatBoostLibraryHandler>(library_path, model_path);
    }
    else
        LOG_WARNING(&Poco::Logger::get("CatBoostLibraryHandlerFactory"), "CatBoost library handler already exists or path to Catboost libary is invalid.");
}

bool CatBoostLibraryHandlerFactory::reset()
{
    std::lock_guard lock(mutex);
    if (library_handler.get())
    {
        /// libDelete is called in destructor.
        library_handler.reset();
        return true;
    }
    return false;
}

}
