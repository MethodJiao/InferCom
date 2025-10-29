#pragma once
/** @class
*  @brief   图形实体符号事件、碰撞检查等
*  @author  北京构力科技有限公司
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2020/5/27
*  ------------------------------------------------------------
*  @note:  -
*/


/** @class
*  @brief   碰撞规则
*  @note:  -
*/
struct ClashRule
{
	vector<BPEntityPtr> m_vecEntity;
	map<PModelId, GeTransform>  m_mapModelTransform;
	ClashRule& operator=(ClashRule rhs)
	{
		this->m_vecEntity.clear();
		this->m_vecEntity = rhs.m_vecEntity;
		this->m_mapModelTransform = rhs.m_mapModelTransform;
		return *this;
	}
};

/** @class
*  @brief   碰撞方法
*  @note:  -
*/
typedef vector<pair<BPEntityPtr, BPEntityPtr>>  ClashResult;
class ClashMethod
{
public:
	ClashMethod();
	~ClashMethod();

	bool doClash(ClashRule const&);
	void getClashResult(ClashResult&);

private:
	void __preFilter();
	void __runClashDetection();

private:
	ClashRule m_rule;
	ClashResult m_clashResult;
	vector<BPEntityPtr> m_vecClashEle;
	vector<PBBim::PBCD::CDObjectNode*> m_vecTrasNode;
};


/** @class
*  @brief   图形实体符号事件
*  @note:  -
*/
class EntitySymbologyEventDemo : public BIMBase::Data::BPEntitySymbologyEventListener
{
protected:
	virtual void _getOverrides(BPSymbologyOverridesR overrids, ::BIMBase::Core::BPEntityCR eh) const;
public:
	static EntitySymbologyEventDemo& Get();
	void begin();
	void end();
	void setSelected(set<BPEntityId>& result);
private:
	bool m_bHaveRegisted;
	CCriticalSection cs;
	set<BPEntityId>  m_selected;
};