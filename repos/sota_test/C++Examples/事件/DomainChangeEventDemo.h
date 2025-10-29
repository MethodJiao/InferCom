#pragma once
/** @class
*  @brief   专业事件
*  @author  北京构力
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2020/5/27
*  ------------------------------------------------------------
*  @note:  -
*/
//更换继承专业切换事件
//如果想要在软件启动时也进事件，需要在项目事件中调用BPDomainEventHandleManager::getInstance()->refreshAll，详见ProjectEventTest
class DomainChangeEventDemo : public BIMBase::Data::IBPDomainEventHandle
{
public:
	virtual bool refreshDomainState(::BIMBase::Core::BPProjectP pPrj );
	virtual bool refreshDomainUi(::BIMBase::Core::BPProjectP pPrj);
	virtual bool initalDomainData(::BIMBase::Core::BPProjectP pPrj);
	virtual bool closedDomainNotify(::BIMBase::Core::BPProjectP pPrj);


};