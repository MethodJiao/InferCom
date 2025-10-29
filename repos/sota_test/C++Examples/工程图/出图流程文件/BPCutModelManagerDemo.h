#pragma once
#include<map>
#include<string>
#include <utility>
#include <vector>


/** @class
 *  @brief   出图模型的管理
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2022/4/26
 *  ------------------------------------------------------------
 *  @note:  -
 */


namespace DemoObject
{

	////定义智能指针、引用等
	class BPCutModelManagerDemo;
	typedef BPCutModelManagerDemo const& BPCutModelManagerDemoCR;
	typedef BPCutModelManagerDemo& BPCutModelManagerDemoR;
	typedef BPCutModelManagerDemo* BPCutModelManagerDemoP;

	class BPCutModelManagerDemo
	{
	public:
		static BPCutModelManagerDemoR Get();
		void addModel(CString modelName, PBModelInfoPtr models);
		std::map<CString, PBModelInfoPtr> getModel() { return m_Namemodels; }
		void deleteModel(PBModelInfoPtr  modelDel);
		GeRange3d  getModelRange(PBModelInfoPtr outModel);

	private:
		std::map<CString, PBModelInfoPtr> m_Namemodels;
		std::vector<PBModelInfoPtr>  m_models;
		std::map<CString , pair<BPModelP, GeRange3d> >m_modelrange;
	};
}
