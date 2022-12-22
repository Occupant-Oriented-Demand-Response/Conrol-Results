within ;
model MultiZone
  parameter Real T0=20 "initial temperature";
  parameter Real sc=2.845 "solar coefficient";
  parameter Real Ra=29.38 * 0.001 "window resistor";
  parameter Real Ri=0.4788 * 0.001 "inside resistor";
  parameter Real Ci=1.183 * 3.6e6 "air capacity";
  parameter Real Cm=3.987 * 3.6e6 "inside capacity";
  parameter Real x1=0.55 "scaling Room1";
  parameter Real x2=0.55 "scaling Room2";
  parameter Real x3=0.55 "scaling Room3";
  parameter Real x4=0.45 "scaling Room4";
  parameter Real x5=0.40 "scaling Room5";
  parameter Real sc_north=0.45 "north orientation";
  parameter Real sc_south=0.55 "south orientation";

  parameter Modelica.Units.SI.CoefficientOfHeatTransfer sc1=sc*sc_south
    "solar coefficient";
  parameter Modelica.Units.SI.Temperature T0i1=T0
    "Initial temperature";
  parameter Modelica.Units.SI.Temperature T0m1=T0
    "Initial temperature";
  parameter Modelica.Units.SI.ThermalResistance Ra1=Ra*x1
    "Constant thermal resistance of material";
  parameter Modelica.Units.SI.ThermalResistance Ri1=Ri*x1
    "Constant thermal resistance of material";
  parameter Modelica.Units.SI.HeatCapacity Ci1=Ci*x1
    "Heat capacity of element (= cp*m)";
  parameter Modelica.Units.SI.HeatCapacity Cm1=Cm*x1
    "Heat capacity of element (= cp*m)";
  parameter Modelica.Units.SI.CoefficientOfHeatTransfer sc2=sc*sc_south
    "solar coefficient";
  parameter Modelica.Units.SI.Temperature T0i2=T0
    "Initial temperature";
  parameter Modelica.Units.SI.Temperature T0m2=T0
    "Initial temperature";
  parameter Modelica.Units.SI.ThermalResistance Ra2=Ra*x2
    "Constant thermal resistance of material";
  parameter Modelica.Units.SI.ThermalResistance Ri2=Ri*x2
    "Constant thermal resistance of material";
  parameter Modelica.Units.SI.HeatCapacity Ci2=Ci*x2
    "Heat capacity of element (= cp*m)";
  parameter Modelica.Units.SI.HeatCapacity Cm2=Cm*x2
    "Heat capacity of element (= cp*m)";
  parameter Modelica.Units.SI.CoefficientOfHeatTransfer sc3=sc*sc_north
    "solar coefficient";
  parameter Modelica.Units.SI.Temperature T0i3=T0
    "Initial temperature";
  parameter Modelica.Units.SI.Temperature T0m3=T0
    "Initial temperature";
  parameter Modelica.Units.SI.ThermalResistance Ra3=Ra*x3
    "Constant thermal resistance of material";
  parameter Modelica.Units.SI.ThermalResistance Ri3=Ri*x3
    "Constant thermal resistance of material";
  parameter Modelica.Units.SI.HeatCapacity Ci3=Ci*x3
    "Heat capacity of element (= cp*m)";
  parameter Modelica.Units.SI.HeatCapacity Cm3=Cm*x3
    "Heat capacity of element (= cp*m)";
  parameter Modelica.Units.SI.CoefficientOfHeatTransfer sc4=sc*sc_north
    "solar coefficient";
  parameter Modelica.Units.SI.Temperature T0i4=T0
    "Initial temperature";
  parameter Modelica.Units.SI.Temperature T0m4=T0
    "Initial temperature";
  parameter Modelica.Units.SI.ThermalResistance Ra4=Ra*x4
    "Constant thermal resistance of material";
  parameter Modelica.Units.SI.ThermalResistance Ri4=Ri*x4
    "Constant thermal resistance of material";
  parameter Modelica.Units.SI.HeatCapacity Ci4=Ci*x4
    "Heat capacity of element (= cp*m)";
  parameter Modelica.Units.SI.HeatCapacity Cm4=Cm*x4
    "Heat capacity of element (= cp*m)";
  parameter Modelica.Units.SI.CoefficientOfHeatTransfer sc5=sc*sc_north
    "solar coefficient";
  parameter Modelica.Units.SI.Temperature T0i5=T0
    "Initial temperature";
  parameter Modelica.Units.SI.Temperature T0m5=T0
    "Initial temperature";
  parameter Modelica.Units.SI.ThermalResistance Ra5=Ra*x5
    "Constant thermal resistance of material";
  parameter Modelica.Units.SI.ThermalResistance Ri5=Ri*x5
    "Constant thermal resistance of material";
  parameter Modelica.Units.SI.HeatCapacity Ci5=Ci*x5
    "Heat capacity of element (= cp*m)";
  parameter Modelica.Units.SI.HeatCapacity Cm5=Cm*x5
    "Heat capacity of element (= cp*m)";


  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Luft1(C=Ci1, T(fixed=true,
        start=T0i1))
    annotation (Placement(transformation(extent={{194,180},{214,200}})));
  Modelica.Blocks.Interfaces.RealOutput Ti1
    annotation (Placement(transformation(extent={{286,140},{306,160}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor Fenster1(R=Ra1)
    annotation (Placement(transformation(extent={{118,170},{138,190}})));
  Modelica.Blocks.Interfaces.RealInput Ta
    annotation (Placement(transformation(extent={{-172,-28},{-132,12}})));
  Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temperatureSensor1
    annotation (Placement(transformation(extent={{244,140},{264,160}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature
    prescribedTemperature1
    annotation (Placement(transformation(extent={{-50,-58},{-30,-38}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Wand1(C=Cm1, T(fixed=true,
        start=T0m1))
    annotation (Placement(transformation(extent={{130,142},{150,162}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor Wand_innen1(R=Ri1)
    annotation (Placement(transformation(extent={{156,124},{176,144}})));
  Modelica.Blocks.Interfaces.RealInput phis
    annotation (Placement(transformation(extent={{-160,118},{-120,158}})));
  Modelica.Blocks.Math.Gain SolarCoeff1(k=sc1)
    annotation (Placement(transformation(extent={{-94,126},{-74,146}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow1
    annotation (Placement(transformation(extent={{-56,128},{-36,148}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Luft2(C=Ci2, T(fixed=true,
        start=T0i2))
    annotation (Placement(transformation(extent={{180,104},{200,124}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor Fenster2(R=Ra2)
    annotation (Placement(transformation(extent={{104,94},{124,114}})));
  Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temperatureSensor2
    annotation (Placement(transformation(extent={{230,64},{250,84}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Wand2(C=Cm2, T(fixed=true,
        start=T0m2))
    annotation (Placement(transformation(extent={{118,66},{138,86}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor Wand_innen2(R=Ri2)
    annotation (Placement(transformation(extent={{142,48},{162,68}})));
  Modelica.Blocks.Interfaces.RealOutput Ti2
    annotation (Placement(transformation(extent={{272,64},{292,84}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow2
    annotation (Placement(transformation(extent={{-54,98},{-34,118}})));
  Modelica.Blocks.Math.Gain SolarCoeff2(k=sc2)
    annotation (Placement(transformation(extent={{-92,98},{-72,118}})));
  Modelica.Blocks.Math.Gain SolarCoeff3(k=sc3)
    annotation (Placement(transformation(extent={{-92,70},{-72,90}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow3
    annotation (Placement(transformation(extent={{-54,70},{-34,90}})));
  Modelica.Blocks.Math.Gain SolarCoeff4(k=sc4)
    annotation (Placement(transformation(extent={{-92,36},{-72,56}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow4
    annotation (Placement(transformation(extent={{-54,36},{-34,56}})));
  Modelica.Blocks.Math.Gain SolarCoeff5(k=sc5)
    annotation (Placement(transformation(extent={{-92,8},{-72,28}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow5
    annotation (Placement(transformation(extent={{-54,8},{-34,28}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor Fenster3(R=Ra3)
    annotation (Placement(transformation(extent={{100,6},{120,26}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Luft3(C=Ci3, T(fixed=true,
        start=T0i3))
    annotation (Placement(transformation(extent={{176,16},{196,36}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Wand3(C=Cm3, T(fixed=true,
        start=T0m3))
    annotation (Placement(transformation(extent={{114,-22},{134,-2}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor Wand_innen3(R=Ri3)
    annotation (Placement(transformation(extent={{138,-40},{158,-20}})));
  Modelica.Blocks.Interfaces.RealOutput Ti3
    annotation (Placement(transformation(extent={{268,-24},{288,-4}})));
  Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temperatureSensor3
    annotation (Placement(transformation(extent={{226,-24},{246,-4}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor Fenster4(R=Ra4)
    annotation (Placement(transformation(extent={{90,-86},{110,-66}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Luft4(C=Ci4, T(fixed=true,
        start=T0i4))
    annotation (Placement(transformation(extent={{166,-76},{186,-56}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Wand4(C=Cm4, T(fixed=true,
        start=T0i4))
    annotation (Placement(transformation(extent={{104,-114},{124,-94}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor Wand_innen4(R=Ri4)
    annotation (Placement(transformation(extent={{128,-132},{148,-112}})));
  Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temperatureSensor4
    annotation (Placement(transformation(extent={{216,-116},{236,-96}})));
  Modelica.Blocks.Interfaces.RealOutput Ti4
    annotation (Placement(transformation(extent={{258,-116},{278,-96}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor Fenster5(R=Ra5)
    annotation (Placement(transformation(extent={{92,-172},{112,-152}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Luft5(C=Ci5, T(fixed=true,
        start=T0i5))
    annotation (Placement(transformation(extent={{168,-162},{188,-142}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Wand5(C=Cm4, T(fixed=true,
        start=T0i5))
    annotation (Placement(transformation(extent={{92,-202},{112,-182}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor Wand_innen5(R=Ri5)
    annotation (Placement(transformation(extent={{130,-218},{150,-198}})));
  Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temperatureSensor5
    annotation (Placement(transformation(extent={{218,-202},{238,-182}})));
  Modelica.Blocks.Interfaces.RealOutput Ti5
    annotation (Placement(transformation(extent={{260,-202},{280,-182}})));
  Modelica.Blocks.Interfaces.RealInput HVAC1
    annotation (Placement(transformation(extent={{220,168},{260,208}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow6
    annotation (Placement(transformation(extent={{270,178},{290,198}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow7
    annotation (Placement(transformation(extent={{280,106},{300,126}})));
  Modelica.Blocks.Interfaces.RealInput HVAC2
    annotation (Placement(transformation(extent={{230,96},{270,136}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow8
    annotation (Placement(transformation(extent={{250,30},{270,50}})));
  Modelica.Blocks.Interfaces.RealInput HVAC3
    annotation (Placement(transformation(extent={{200,20},{240,60}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow9
    annotation (Placement(transformation(extent={{248,-68},{268,-48}})));
  Modelica.Blocks.Interfaces.RealInput HVAC4
    annotation (Placement(transformation(extent={{198,-78},{238,-38}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow10
    annotation (Placement(transformation(extent={{238,-156},{258,-136}})));
  Modelica.Blocks.Interfaces.RealInput HVAC5
    annotation (Placement(transformation(extent={{188,-166},{228,-126}})));
equation
  connect(Fenster1.port_b, Luft1.port)
    annotation (Line(points={{138,180},{204,180}},
                                                color={191,0,0}));
  connect(Luft1.port, temperatureSensor1.port)
    annotation (Line(points={{204,180},{204,150},{244,150}},color={191,0,0}));
  connect(Ti1, temperatureSensor1.T)
    annotation (Line(points={{296,150},{265,150}}, color={0,0,127}));
  connect(Ta, prescribedTemperature1.T) annotation (Line(points={{-152,-8},{-62,
          -8},{-62,-48},{-52,-48}}, color={0,0,127}));
  connect(Ti1, Ti1)
    annotation (Line(points={{296,150},{296,150}}, color={0,0,127}));
  connect(Wand_innen1.port_a, Wand1.port)
    annotation (Line(points={{156,134},{140,134},{140,142}},
                                                           color={191,0,0}));
  connect(Wand_innen1.port_b, Luft1.port)
    annotation (Line(points={{176,134},{204,134},{204,180}},color={191,0,0}));
  connect(SolarCoeff1.u, phis) annotation (Line(points={{-96,136},{-118,136},{-118,
          138},{-140,138}}, color={0,0,127}));
  connect(prescribedHeatFlow1.Q_flow, SolarCoeff1.y)
    annotation (Line(points={{-56,138},{-64,138},{-64,136},{-73,136}},
                                                 color={0,0,127}));
  connect(prescribedHeatFlow1.port, Luft1.port) annotation (Line(points={{-36,138},
          {150,138},{150,180},{204,180}},
                                       color={191,0,0}));
  connect(Fenster2.port_b, Luft2.port)
    annotation (Line(points={{124,104},{190,104}},color={191,0,0}));
  connect(Luft2.port,temperatureSensor2. port) annotation (Line(points={{190,104},
          {190,74},{230,74}},     color={191,0,0}));
  connect(Ti2, temperatureSensor2.T)
    annotation (Line(points={{282,74},{251,74}}, color={0,0,127}));
  connect(Ti2, Ti2)
    annotation (Line(points={{282,74},{282,74}}, color={0,0,127}));
  connect(Wand_innen2.port_a, Wand2.port)
    annotation (Line(points={{142,58},{128,58},{128,66}},    color={191,0,0}));
  connect(Wand_innen2.port_b, Luft2.port) annotation (Line(points={{162,58},{190,
          58},{190,104}},   color={191,0,0}));
  connect(SolarCoeff2.u, phis) annotation (Line(points={{-94,108},{-104,108},{-104,
          138},{-140,138}}, color={0,0,127}));
  connect(prescribedHeatFlow2.Q_flow, SolarCoeff2.y)
    annotation (Line(points={{-54,108},{-71,108}},
                                                 color={0,0,127}));
  connect(prescribedHeatFlow2.port, Luft2.port) annotation (Line(points={{-34,108},
          {200,108},{200,104},{190,104}},
                                       color={191,0,0}));
  connect(prescribedHeatFlow3.Q_flow, SolarCoeff3.y)
    annotation (Line(points={{-54,80},{-71,80}}, color={0,0,127}));
  connect(prescribedHeatFlow4.Q_flow, SolarCoeff4.y)
    annotation (Line(points={{-54,46},{-71,46}}, color={0,0,127}));
  connect(prescribedHeatFlow5.Q_flow, SolarCoeff5.y)
    annotation (Line(points={{-54,18},{-71,18}}, color={0,0,127}));
  connect(Luft3.port,temperatureSensor3. port) annotation (Line(points={{186,16},
          {186,-14},{226,-14}},   color={191,0,0}));
  connect(Wand_innen3.port_b,Luft3. port) annotation (Line(points={{158,-30},{186,
          -30},{186,16}},   color={191,0,0}));
  connect(Wand_innen3.port_a,Wand3. port)
    annotation (Line(points={{138,-30},{124,-30},{124,-22}}, color={191,0,0}));
  connect(Fenster3.port_b,Luft3. port)
    annotation (Line(points={{120,16},{186,16}},  color={191,0,0}));
  connect(Ti3, temperatureSensor3.T)
    annotation (Line(points={{278,-14},{247,-14}}, color={0,0,127}));
  connect(Luft4.port,temperatureSensor4. port) annotation (Line(points={{176,-76},
          {176,-106},{216,-106}}, color={191,0,0}));
  connect(Wand_innen4.port_b,Luft4. port) annotation (Line(points={{148,-122},{176,
          -122},{176,-76}}, color={191,0,0}));
  connect(Wand_innen4.port_a,Wand4. port)
    annotation (Line(points={{128,-122},{114,-122},{114,-114}},
                                                             color={191,0,0}));
  connect(Fenster4.port_b,Luft4. port)
    annotation (Line(points={{110,-76},{176,-76}},color={191,0,0}));
  connect(Ti4, temperatureSensor4.T)
    annotation (Line(points={{268,-106},{237,-106}}, color={0,0,127}));
  connect(Luft5.port,temperatureSensor5. port) annotation (Line(points={{178,-162},
          {178,-192},{218,-192}}, color={191,0,0}));
  connect(Wand_innen5.port_b,Luft5. port) annotation (Line(points={{150,-208},{178,
          -208},{178,-162}},color={191,0,0}));
  connect(Wand_innen5.port_a,Wand5. port)
    annotation (Line(points={{130,-208},{102,-208},{102,-202}},
                                                             color={191,0,0}));
  connect(Fenster5.port_b,Luft5. port)
    annotation (Line(points={{112,-162},{178,-162}},
                                                  color={191,0,0}));
  connect(Ti5, temperatureSensor5.T)
    annotation (Line(points={{270,-192},{239,-192}}, color={0,0,127}));
  connect(prescribedHeatFlow3.port, Luft3.port) annotation (Line(points={{-34,80},
          {144,80},{144,16},{186,16}}, color={191,0,0}));
  connect(prescribedHeatFlow4.port, Luft4.port) annotation (Line(points={{-34,46},
          {136,46},{136,-76},{176,-76}}, color={191,0,0}));
  connect(SolarCoeff3.u, phis) annotation (Line(points={{-94,80},{-102,80},{-102,
          138},{-140,138}}, color={0,0,127}));
  connect(SolarCoeff4.u, phis) annotation (Line(points={{-94,46},{-112,46},{-112,
          138},{-140,138}}, color={0,0,127}));
  connect(SolarCoeff5.u, phis) annotation (Line(points={{-94,18},{-106,18},{-106,
          138},{-140,138}}, color={0,0,127}));
  connect(prescribedHeatFlow6.Q_flow, HVAC1)
    annotation (Line(points={{270,188},{240,188}}, color={0,0,127}));
  connect(prescribedHeatFlow7.Q_flow, HVAC2)
    annotation (Line(points={{280,116},{250,116}}, color={0,0,127}));
  connect(prescribedHeatFlow8.Q_flow, HVAC3)
    annotation (Line(points={{250,40},{220,40}}, color={0,0,127}));
  connect(prescribedHeatFlow9.Q_flow, HVAC4)
    annotation (Line(points={{248,-58},{218,-58}}, color={0,0,127}));
  connect(prescribedHeatFlow10.Q_flow, HVAC5)
    annotation (Line(points={{238,-146},{208,-146}}, color={0,0,127}));
  connect(prescribedHeatFlow10.port, Luft5.port) annotation (Line(points={{258,
          -146},{264,-146},{264,-172},{178,-172},{178,-162}}, color={191,0,0}));
  connect(prescribedHeatFlow9.port, Luft4.port) annotation (Line(points={{268,
          -58},{274,-58},{274,-86},{176,-86},{176,-76}}, color={191,0,0}));
  connect(prescribedHeatFlow8.port, Luft3.port) annotation (Line(points={{270,
          40},{276,40},{276,6},{186,6},{186,16}}, color={191,0,0}));
  connect(prescribedHeatFlow7.port, temperatureSensor2.port) annotation (Line(
        points={{300,116},{306,116},{306,98},{190,98},{190,74},{230,74}}, color=
         {191,0,0}));
  connect(prescribedHeatFlow6.port, temperatureSensor1.port) annotation (Line(
        points={{290,188},{306,188},{306,170},{204,170},{204,150},{244,150}},
        color={191,0,0}));
  connect(Fenster1.port_a, prescribedTemperature1.port) annotation (Line(points=
         {{118,180},{70,180},{70,120},{54,120},{54,-48},{-30,-48}}, color={191,0,
          0}));
  connect(Fenster2.port_a, prescribedTemperature1.port) annotation (Line(points=
         {{104,104},{54,104},{54,-48},{-30,-48}}, color={191,0,0}));
  connect(Fenster4.port_a, prescribedTemperature1.port) annotation (Line(points=
         {{90,-76},{72,-76},{72,-48},{-30,-48}}, color={191,0,0}));
  connect(Fenster3.port_a, prescribedTemperature1.port) annotation (Line(points=
         {{100,16},{54,16},{54,-48},{-30,-48}}, color={191,0,0}));
  connect(Fenster5.port_a, prescribedTemperature1.port) annotation (Line(points=
         {{92,-162},{54,-162},{54,-48},{-30,-48}}, color={191,0,0}));
  connect(prescribedHeatFlow5.port, Luft5.port) annotation (Line(points={{-34,18},
          {152,18},{152,-162},{178,-162}}, color={191,0,0}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false, extent={{-160,-240},
            {320,220}})),                                        Diagram(
        coordinateSystem(preserveAspectRatio=false, extent={{-160,-240},{320,220}})),
    uses(Modelica(version="4.0.0")));
end MultiZone;
