#pragma once
/** @class
*  @brief   立方体造型
*  @author  北京构力科技有限公司
*  @date    2022/4/19
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2022/4/19
*  ------------------------------------------------------------
*  @note:  存的是局部坐标系下的信息
*/

namespace DemoObject
{
	//定义智能指针、引用等
	class ModeArchWallDemo;
	typedef ModeArchWallDemo const& ModeArchWallDemoCR;
	typedef ModeArchWallDemo& ModeArchWallDemoR;
	typedef ModeArchWallDemo* ModeArchWallDemoP;
	typedef RefCountedPtr<ModeArchWallDemo>  ModeArchWallDemoPtr;

	class ModeArchWallDemo
		: public CubeDemo
	{

	public:
		ModeArchWallDemo();
		~ModeArchWallDemo();

	protected:
		virtual Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_Demo; };
		virtual Utf8String	 _getClassName() const override { return PBM_CLASS_ARCH_WALL_Demo; };
		//创建几何造型
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool bIsDynamics) override;

		Demo_CREATE(ModeArchWallDemo);
	};
	Demo_EXTENSION(ModeArchWallDemo);
}

