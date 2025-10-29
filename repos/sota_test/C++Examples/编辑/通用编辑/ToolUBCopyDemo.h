#pragma once
/** @class
 *  @brief   工字钢复制工具
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/5/11
 *  ------------------------------------------------------------
 *  @note:  -
 */

class ToolUBCopyDemo :public IToolCopy
{

    virtual void ElementsSelected(std::vector<::BIMBase::Core::BPEntityPtr>& refps);
    virtual void Dynamic(std::vector<::BIMBase::Core::BPEntityPtr> const& refps, p3d::GeTransformCR transform, ::BIMBase::Core::BPRedrawEntitys& redrawEntitys);
    virtual void Copy(std::vector<::BIMBase::Core::BPEntityPtr> const& refps, p3d::GeTransformCR transform, unsigned int ncopy);
    virtual bool CacheDynamic() { return true; };
};

