#pragma once
/** @class
*  @brief   右键菜单事件
*  @author  北京构力
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2022/5/27
*  ------------------------------------------------------------
*  @note:  -
*/
//想要定制自己的右键菜单，需要继承事件，重写接口编辑右键菜单。然后将该类注册到右键菜单事件消息管理类BPViewRButtonClickListenerCenter
class RButtonClickEventDemo : public BIMBase::FrameWork::BPViewRButtonClickListener
{
protected:
	virtual void _setMenu(BIMBase::FrameWork::RButtonClickItemPtr& initMenu);
};

