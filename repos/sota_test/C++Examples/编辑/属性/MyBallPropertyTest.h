#pragma once
/** @class
 *  @brief   球体属性页
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/4/26
 *  ------------------------------------------------------------
 *  @note:  本属性方法实现了自定义属性功能,可配合自定义属性管理器使用
 */

class MyBallPropertyTest :public wd::WDToolProperty
{
	DefineSuper(wd::WDToolProperty)
	enum BallPropName
	{
		OriginX,         //圆心X
		OriginY,         //圆心Y
		OriginZ,         //圆心Z
		BallPropCount
	};

public:
	MyBallPropertyTest();
	~MyBallPropertyTest();

	//获取属性并且在属性框显示
	virtual void OnPropertyGet(std::vector<BPEntityP> const& refps, PBBimUIProperyList& lst)  override;
	//设置属性框中的值
	virtual TIErrorStatus OnPropertySet(std::vector<BPEntityP> const& refps, int index, PBBimUIPropertyItem const& item) override;

	virtual void OnPropertyGet(::BIMBase::Data::IBPObjectPtr& ptrObj, PBBimUIProperyList& lst) { return; };
	virtual TIErrorStatus OnPropertySet(::BIMBase::Data::IBPObjectPtr& ptrObj, int index, PBBimUIPropertyItem const& item) { return TIErrorStatus::succeed; };
};

