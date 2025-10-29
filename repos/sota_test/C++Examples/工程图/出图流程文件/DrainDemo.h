#pragma once
/** @class
*  @brief   排水沟造型
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
	class DrainDemo;
	typedef DrainDemo const& DrainDemoCR;
	typedef DrainDemo& DrainDemoR;
	typedef DrainDemo* DrainDemoP;

	class DrainDemo
		: public DemoObject::BPDemoGraphicElementDemo
	{
		DefineSuper(BPDemoGraphicElementDemo)

	public:
		DrainDemo();
		~DrainDemo();

		int getWidth() const;
		void   setWidth(int nWidth);

		int getLength() const;
		void   setLength(int nLength);


		int getThickness() const;
		void   setThickness(int nThickness);

		int getDepth() const;
		void   setDepth(int nDepth);
		//BIMBase::Core::BPGraphicsPtr createPhysicalGraphicsForDrawing(BIMBase::Core::BPProject& project, BIMBase::PModelIdCR modelId);
	protected:
		virtual Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_Demo; };
		virtual Utf8String	 _getClassName() const override { return "DrainDemo"; };
		//写数据
		virtual ::p3d::P3DStatus _copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const override;
		//读数据
		virtual ::p3d::P3DStatus _initFromData(BIMBase::Core::BPDataCR	instance) override;
		//创建几何造型
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphicsForDrawing(BIMBase::Core::BPProject& project, BIMBase::PModelIdCR modelId) override;
		//virtual ::p3d::P3DStatus                _addToProject(::BIMBase::Core::BPProjectR project, ::BIMBase::PModelIdCR modelId) override;
		//创建几何造型
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool bIsDynamics) override;
	private:
		int m_nWidth;
		int m_nThickness;
		int m_nDepth;
		int m_nLenght;

		Demo_CREATE(DrainDemo);
	};
	Demo_EXTENSION(DrainDemo);
}

