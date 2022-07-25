#pragma once

#include <Interpreters/Context.h>
#include <Bridge/IBridge.h>
#include "CatBoostBridgeHandlerFactory.h"

namespace DB
{

class CatBoostBridge : public IBridge
{
protected:
    std::string bridgeName() const override;
    HandlerFactoryPtr getHandlerFactoryPtr(ContextPtr context) const override;
};

}
