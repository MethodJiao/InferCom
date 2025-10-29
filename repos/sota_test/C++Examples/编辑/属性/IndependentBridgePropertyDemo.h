#pragma once



	class IndependentBridgePropertyDemo :public IToolProperty
	{
		enum IBPropName
		{
			enPATTERN,
			enCOLUMNDIAMETER,
			enCOLUMNHEIGHT,
			enARCHHEIGHT,
			enROWS,
			enCOLUMNS,
			enCSSLENGHT,
			enCSSWIDTH,
			enCSSHEIGHT,
			enTOPSLABTHICKNESS,
			enSIDESLABTHICKNESS,
			enTUBEDIAMETER,
			enTUBETHICKNESS,
			enPROPCOUNT
		};

		
	public:
		//获取属性并且在属性框显示
		virtual void OnPropertyGet(std::vector<BPEntityP> const& refps, PBBimUIProperyList& lst)  override;
		//设置属性框中的值
		virtual TIErrorStatus OnPropertySet(std::vector<BPEntityP> const& refps, int index, PBBimUIPropertyItem const& item) override;
	
	private:
		std::vector<double> __wstring2doubles(wstring);
		wstring __doubles2wstring(vector<double>);
	};
