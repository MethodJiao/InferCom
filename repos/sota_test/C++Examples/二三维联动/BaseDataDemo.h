#pragma once
/** @class
 *  @brief   二进制存取范例里用到的线，体的基类，没有用schema存取数据
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
	//定义智能指针、引用等
	class BaseDataDemo;
	typedef BaseDataDemo const& BaseDataDemoCR;
	typedef BaseDataDemo& BaseDataDemoR;
	typedef BaseDataDemo* BaseDataDemoP;
	
	class BaseDataDemo
	{
		public:
			BaseDataDemo() {};
			~BaseDataDemo(){};
			virtual BIMBase::Core::BPGraphicsPtr createGraphics() { return nullptr; };
			virtual CString getClassName() { return L""; };
	};
}