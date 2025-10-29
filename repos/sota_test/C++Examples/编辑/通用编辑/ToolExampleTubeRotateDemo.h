#pragma once
/** @class
 *  @brief   圆管旋转工具
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/5/11
 *  ------------------------------------------------------------
 *  @note:  -
 */


class ToolExampleTubeRotate : public  IToolRotate
{
	virtual void ElementsSelected(std::vector<::BIMBase::Core::BPEntityPtr>& refps) {};
	virtual void Dynamic(std::vector<::BIMBase::Core::BPEntityPtr> const& refps, GeTransformCR transform, ::BIMBase::Core::BPRedrawEntitys& redrawElems) {};
	virtual void Rotate(std::vector<::BIMBase::Core::BPEntityPtr> const& refps, GeTransformCR transform);
	virtual bool CacheDynamic() { return false; };
};


