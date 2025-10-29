#pragma once
/** @class
 *  @brief   圆管删除工具
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/5/11
 *  ------------------------------------------------------------
 *  @note:  -
 */

class ToolExampleTubeDelete :public IToolDelete
{

    virtual void ElementsSelected(std::vector<::BIMBase::Core::BPEntityPtr>& refps);

    virtual TIErrorStatus Delete(std::vector<::BIMBase::Core::BPEntityPtr> const& refps);
};

