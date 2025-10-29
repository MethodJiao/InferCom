#pragma once
/** @class
*  @brief   路堤造型
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
	class EmbankmentDemo;
	typedef EmbankmentDemo const& EmbankmentDemoCR;
	typedef EmbankmentDemo& EmbankmentDemoR;
	typedef EmbankmentDemo* EmbankmentDemoP;

	class EmbankmentDemo
	: public  DemoObject::BPDemoGraphicElementDemo
	{
		DefineSuper(BPDemoGraphicElementDemo)

	public:
		EmbankmentDemo();
		~EmbankmentDemo();

		int getTopWidth() const;
		void   setTopWidth(int nWidth);

		int getLength() const;
		void   setLength(int nLength);

		double getSlop() const;
		void   setSlop(double nHeight);

		int getGravelThickness() const;
		void   setGravelThickness(int nHeight);

		int getPackingThickness() const;
		void   setPackingThickness(int nHeight);
	protected:
		virtual Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_Demo; };
		virtual Utf8String	 _getClassName() const override { return "EmbankmentDemo"; };
		//写数据
		virtual ::p3d::P3DStatus _copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const override;
		//读数据
		virtual ::p3d::P3DStatus _initFromData(BIMBase::Core::BPDataCR	instance) override;
		//创建几何造型
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool bIsDynamics) override;
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphicsForDrawing(BIMBase::Core::BPProject& project, BIMBase::PModelIdCR modelId) override { return nullptr; }

	private:
		int m_nTopWidth;
		int m_nGravelThickness;
		int m_nPackingThickness;
		int m_nLenght;
		double m_dSlop;

		Demo_CREATE(EmbankmentDemo);
	};
	Demo_EXTENSION(EmbankmentDemo);
}

