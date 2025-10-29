#pragma once
#include "PBBimTools/IToolMove.h"
/** @class
 *  @brief   排水沟移动工具
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/5/11
 *  ------------------------------------------------------------
 *  @note:  -
 */
class ToolDrainMoveDemo :public IToolMove
{
public:
	ToolDrainMoveDemo();
	~ToolDrainMoveDemo();

	//选中需要移动的构件后响应
	virtual void ElementsSelected(std::vector<BPEntityPtr>& refps) override;
	//如果返回false则会调用自定义Dynamic函数，调自定义函数时最好不要是计算密集型操作
	virtual bool CacheDynamic() override { return true; }
	//动态移动物体时响应函数，需要CacheDynamic为false时函数启动
	virtual void Dynamic(std::vector<BPEntityPtr> const& refps, GeTransformCR transform, BPRedrawEntitys& redrawElems) override;
	//移动完构件点击布置时响应函数
	virtual void Move(std::vector<BPEntityPtr> const& refps, GeTransformCR transform) override;
};

