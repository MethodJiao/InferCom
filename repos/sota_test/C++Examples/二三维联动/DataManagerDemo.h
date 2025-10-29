#pragma once
/** @class
 *  @brief   二进制存取范例里数据管理类，存在schema的二进制块里
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
	//定义智能指针、引用等
	class DataManagerDemo;
	typedef DataManagerDemo const& DataManagerDemoCR;
	typedef DataManagerDemo& DataManagerDemoR;
	typedef DataManagerDemo* DataManagerDemoP;
	typedef RefCountedPtr<DataManagerDemo> DataManagerDemoPtr;

	class DataManagerDemo
		: public BIMBase::Data::BPGraphicElement
	{
		DefineSuper(BPGraphicElement)

	public:
		DataManagerDemo();
		~DataManagerDemo();
		static DataManagerDemoR Get();
		bool addPhysicalGraphics(BIMBase::Core::BPProjectR project, DemoObject::BaseDataDemoP pBasedata);
		BIMBase::Core::BPGraphicsPtr getPhysicalGraphics(BIMBase::Core::BPProjectR project);
		void setByte(byte* byte) { m_byte = byte; }
		byte* getByte() { return m_byte; }
		pvector<byte*> getAllByte();
		
		void setStringLength(size_t strLen) { m_nStrLen = strLen; }
		int getStringLength()const { return m_nStrLen; }

		void setByteLength(size_t byteLen) { m_nByteLen = byteLen; }
		int getByteLength() const { return m_nByteLen; }

		void setStringVec(pvector<int> strvec) { m_nStrvec = strvec; }
		pvector<int>  getStringVec() const { return m_nStrvec; }

		void setByteVec(pvector<int> strvec) { m_nStrvec = strvec; }
		pvector<int>  getByteVec() const { return m_nBytevec; }

		void setClassName(CString classname) { m_classname = classname; }
		CString getClassName() const { return m_classname; }

		void setPts(std::vector<GePoint3d> pt) { m_pts = pt; }
		std::vector<GePoint3d> getPts()const { return m_pts; }

		bool serialize_Cereal(std::vector<char>& binaryDataVec)const;
		bool deSerialize_Cereal(std::vector<char>& binaryDataVec);
	
	protected:
		virtual Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_Demo; };
		virtual Utf8String	 _getClassName() const override { return PBM_CLASS_DATAMANAGER_Demo; };
		//写数据
		virtual ::p3d::P3DStatus _copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const override;
		//读数据
		virtual ::p3d::P3DStatus _initFromData(BIMBase::Core::BPDataCR	instance) override;
		//创建几何造型
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool isDynamics) override;
		
		//用cereal方法去序列化数据
		
		virtual bool _serialize_Cereal(cereal::BinaryOutputArchive& ar)const ;//需要具体使用类实现
		virtual bool _deSerialize_Cereal(cereal::BinaryInputArchive& ar);
		
	private:
		byte* m_byte;
		int m_nVersion;
		int m_nByteLen;
		int m_nStrLen;
		pvector<int> m_nStrvec;
		pvector<int> m_nBytevec;
		std::vector<GePoint3d>m_pts;
		CString m_classname;

		Demo_CREATE(DataManagerDemo);
	};
	 Demo_EXTENSION(DataManagerDemo);
}
