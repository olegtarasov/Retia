<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:ms="http://schemas.microsoft.com/developer/msbuild/2003">

    <xsl:output method="xml" version="1.0" encoding="UTF-8" indent="yes"/>

    <!-- Identity. -->
    <xsl:template match="@* | node()">
      <xsl:copy>
        <xsl:apply-templates select="@* | node()" />
      </xsl:copy>
    </xsl:template>

    <xsl:template match="//ms:Import[@Project='NugetRelativeRefs.targets']">
    </xsl:template>

    <xsl:template match="//ms:HintPath[contains(text(), '..\packages\')]">
      <xsl:param name="pText" select="substring-after(text(), '..\packages\')"/>
      <xsl:element namespace="http://schemas.microsoft.com/developer/msbuild/2003" name="HintPath">
        <xsl:value-of select="concat('$(SolutionDir)\packages\', $pText)" />
      </xsl:element>
    </xsl:template>

</xsl:stylesheet>
