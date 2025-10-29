#pragma once
/** @class  
 *  @brief   工程响应事件
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/5/11
 *  ------------------------------------------------------------
 *  @note:  -  
 */

class ProjectEventDemo : public BPProjectEventListener
{
protected:
	virtual bool    _onPreOpen(const ProjectPreOpenArg& arg);
	virtual bool    _onPostOpen(::BIMBase::Core::BPProjectR project);

	virtual bool    _onPreClose(::BIMBase::Core::BPProjectR project);
	virtual bool    _onPostClose(::BIMBase::Core::BPProjectR project);
};

