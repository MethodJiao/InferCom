#pragma once
/** @classOpenningDemo
*  @brief   球造型
*  @author  北京构力科技有限公司
*  @date    2022/4/19
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2022/4/19
*  ------------------------------------------------------------
*  @note: 存的是世界坐标系下的点，一般情况下造型存局部坐标系下的点，具体可参考cubeDemo
*/

namespace DemoObject
{ 
	//定义智能指针、引用等
	class OpenningDemo;
	typedef OpenningDemo const&          OpenningDemoCR;
	typedef OpenningDemo&                OpenningDemoR;
	typedef OpenningDemo*                OpenningDemoP;
	typedef RefCountedPtr<OpenningDemo>  OpenningDemoPtr;

	class OpenningDemo
	: public BIMBase::Data::BPGraphicElement
	{
		DefineSuper(BPGraphicElement)

	public:
		OpenningDemo();
		~OpenningDemo();

		int getWidth() const;
		void   setWidth(int nWidth);

		int getLength() const;
		void   setLength(int nLength);

		int getHeight() const;
		void   setHeight(int nHeight);

	protected:
		virtual Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_Demo; };
		virtual Utf8String	 _getClassName() const override { return PBM_CLASS_OPENNING_Demo; };
		virtual ::p3d::P3DStatus      _copyToData(::BIMBase::Core::BPDataR data, ::BIMBase::Core::BPProjectR project) const override;
		virtual ::p3d::P3DStatus      _initFromData(::BIMBase::Core::BPDataCR data) override;
		virtual BIMBase::Core::BPGraphicsPtr    _createPhysicalGraphics(::BIMBase::Core::BPProjectR project, ::BIMBase::PModelIdCR modelId, bool isDynamics) override;
	private:
		int m_nWidth;
		int m_nHeight;
		int m_nLenght;

		Demo_CREATE(OpenningDemo);
	};
	Demo_EXTENSION(OpenningDemo);
}

