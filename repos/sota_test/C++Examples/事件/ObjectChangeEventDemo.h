#pragma once
/** @class
*  @brief   对象处理事件
*  @author  北京构力
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2021/5/11
*  ------------------------------------------------------------
*  @note:  -
*/
class ObjectChangeEventDemo : public BPObjectChangeEventListener
{
public:
	ObjectChangeEventDemo();
	~ObjectChangeEventDemo();

protected:
	virtual bool _onPostNew(IBPObjectCR arg) override;
	virtual bool    _onPostEdit(IBPObjectCR arg) override;
	virtual bool    _onPreDelete(IBPObjectCR pbObject) override;
};

