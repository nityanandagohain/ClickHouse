#include "CatBoostBridge.h"

int main(int argc, char ** argv)
{
    DB::CatBoostBridge app;
    try
    {
        return app.run(argc, argv);
    }
    catch (...)
    {
        std::cerr << DB::getCurrentExceptionMessage(true) << "\n";
        auto code = DB::getCurrentExceptionCode();
        return code ? code : 1;
    }
}

namespace DB
{

std::string CatBoostBridge::bridgeName() const
{
    return "CatBoostBridge";
}

CatBoostBridge::HandlerFactoryPtr CatBoostBridge::getHandlerFactoryPtr(ContextPtr context) const
{
    return std::make_shared<CatBoostBridgeHandlerFactory>("CatBoostRequestHandlerFactory", keep_alive_timeout, context);
}

}
