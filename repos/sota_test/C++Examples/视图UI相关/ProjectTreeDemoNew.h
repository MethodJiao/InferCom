#pragma once

/** @class
*  @brief   新创建项目树方法
*  @author  北京构力
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2023/11/16
*  ------------------------------------------------------------
*  @note:  -
*/

class ProjectTreeDemoNew:public BIMBase::ProjectTree::BPProjectTree
{
protected:
	/**
	@brief		节点增加前
	@param[in]	pNode 增加的节点
	*/
	virtual void _preAdd(IN BIMBase::Data::BPTreeNodeP pNode);
	/**
	@brief		节点增加后
	@param[in]	pNode 增加的节点
	@param[in]	bSelect 是否选中-仅对当前激活项目树增加时有效
	*/
	virtual void _posAdd(IN BIMBase::Data::BPTreeNodeP pNode, IN bool bSelect = false);
	/**
	@brief		节点删除前
	@param[in]	pNode 删除的节点
	*/
	virtual void _preDelete(IN BIMBase::Data::BPTreeNodeP pNode);
	/**
	@brief		节点删除后
	@param[in]	pNode 删除的节点
	*/
	virtual void _posDelete(IN BIMBase::Data::BPTreeNodeP pNode) {};
	/**
	@brief		节点重命名前
	@param[in]	pNode 重命名的节点
	@param[in]	wsNewName 重命名的新名字
	*/
	virtual void _preRename(IN BIMBase::Data::BPTreeNodeP pNode, IN std::wstring wsNewName) {};
	/**
	@brief		节点重命名后
	@param[in]	pNode 重命名的节点
	@param[in]	wsNewName 重命名前的旧名字
	*/
	virtual void _posRename(IN BIMBase::Data::BPTreeNodeP pNode, IN std::wstring wsOldName) {};
	/**
	@brief		双击项目树节点行为（UI）
	@param[in]	pNode 双击的节点
	*/
	virtual void _onDbClick(IN BIMBase::Data::BPTreeNodeP pNode);
	/**
	@brief		左键项目树节点行为（UI）
	@param[in]	pNode 左键的节点
	*/
	virtual void _onLeftClick(IN BIMBase::Data::BPTreeNodeP pNode) {};
	/**
	@brief		右键项目树节点行为（UI）
	@param[in]	pNode 右键的节点
	*/
	virtual void _onRightClick(IN BIMBase::Data::BPTreeNodeP pNode);
	/**
	@brief		激活时获取树节点用图标
	@return		CImageList* 图标
	*/
	virtual CImageList* _getIconImage() { return nullptr; }
};

