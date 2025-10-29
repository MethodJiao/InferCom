#pragma once
/** @class  
 *  @brief   立方体属性页
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/5/15
 *  ------------------------------------------------------------
 *  @note: 标准的属性面板实现机制  
 */

class MyCubePropertyDemo :public IToolProperty
{
	enum CubePropName
	{
		Length,         //长度
		Height,         //高度
		Width,          //宽度
		ExtendProperty, //扩展属性
		CubePropCount
	};

public:
	MyCubePropertyDemo();
	~MyCubePropertyDemo();

	//获取属性并且在属性框显示
	virtual void OnPropertyGet(std::vector<BPEntityP> const & refps, PBBimUIProperyList& lst)  override;
	//设置属性框中的值
	virtual TIErrorStatus OnPropertySet(std::vector<BPEntityP> const & refps, int index, PBBimUIPropertyItem const & item) override;
};

