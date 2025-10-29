#pragma once
/** @class
 *  @brief   二进制存取范例里用到的线，体的类，没有用schema存取数据
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
	class BaseDataDemo;
	class LineDemo;
	typedef LineDemo const& LineDemoCR;
	typedef LineDemo& LineDemoR;
	typedef LineDemo* LineDemoP;
	class SoildCubeDemo;
	typedef SoildCubeDemo const& SoildCubeDemoCR;
	typedef SoildCubeDemo& SoildCubeDemoR;
	typedef SoildCubeDemo* SoildCubeDemoP;

	class LineDemo : public BaseDataDemo

	{


	public:
		LineDemo();
		~LineDemo();
		virtual BIMBase::Core::BPGraphicsPtr createGraphics() override;
		virtual CString getClassName() override { return L"LineDemo"; };
		BIMBase::Core::BPGraphicsPtr m_ptrLineGraphics;
	
		
	};
	class SoildCubeDemo : public BaseDataDemo

	{


	public:
		SoildCubeDemo();
		~SoildCubeDemo();
		virtual BIMBase::Core::BPGraphicsPtr createGraphics() override;
		virtual CString getClassName() override { return L"SoildCubeDemo"; };
		BIMBase::Core::BPGraphicsPtr m_ptrSoildGraphics;


	};


}
