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
class ElementChangeEventDemo : public BPEntityChangeEventListener
{
public:
	ElementChangeEventDemo();
	~ElementChangeEventDemo();

protected:
	virtual bool _onPostNew(BPEntityChangeEventArgCR arg) override;
	virtual bool    _onPostEdit(BPEntityChangeEventArgCR arg) override;
	virtual bool    _onPreDelete(BPEntityChangeEventArgCR arg) override;
};

